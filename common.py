import logging
import torch
import torch.nn as nn
from cvpods.modeling.xkmodel import MLP, SoftmaxPredictor
from cvpods.utils.xklib import stack_data_from_batch
import torchvision.models as models
from cvpods import model_zoo

def get_raw_resnet(logits_type):
    print ("Get new resnet.")
    resnet = models.resnet18(pretrained=True)
    return nn.Sequential(*list(resnet.children())[:-2]) # output is 512

model_type_config = {
    'widenet': {
        'out_dim': 640,
        'model_zoo_path': '/home/data/GAN/gan-based/widenet/',
        'weights': "",
    }, 
    'resnet18': {
        'out_dim': 512, 
        'model_zoo_path': '/home/data/GAN/gan-based/resnet18/',
        'weights': "/home/data/Output/resnet18-cifar/model_final_best.pth",
    },
}

def get_frozen_backbone(logits_type, model_type):
    """
    NOTE : the output of resnet is 512, because we cut off the 
           average pull and fc.

    This method return a resnet as pretrained model, and 
    frozen the parameters of this resnet.
    """
    print ("Get frozen resnet.")
    assert model_type in ['widenet', 'resnet18']
    custom_config = dict(
        MODEL=dict(
            WEIGHTS=model_type_config[model_type]['weights'],
        ),
    )
    baseline = model_zoo.get(
        model_type_config[model_type]['model_zoo_path'],
        trained=False,
        custom_config=custom_config,
    )
    for name, param in baseline.named_parameters():
        param.required_grad = False
    baseline.eval()
    if logits_type == "DEFAULT":
        return baseline.backbone
    elif logits_type == "LABEL_ONLY":
        return baseline

def get_mlp(cfg, logits_type, model_type):
    output_dim = model_type_config[model_type]['out_dim']
    if logits_type == 'LABEL_ONLY':
        mlp = MLP([2 * cfg.MODEL.DIM, 2 * cfg.MODEL.DIM, 1], torch.nn.Tanh, False)
    elif logits_type == "DEFAULT": 
        mlp = MLP([output_dim + cfg.MODEL.DIM, 32, 1], torch.nn.Tanh, False)
    return mlp

class ModelBase(nn.Module):
    def _get_backbone(self, type, model_type):
        return get_frozen_backbone(type, model_type)

    def set_backbone(self, backbone):
        if self.logits_type == 'LABEL_ONLY':
            self.resnet = backbone
        else : 
            self.resnet = backbone.backbone

        for name, param in backbone.named_parameters():
            param.required_grad = False

    def reset_backbone(self, backbone):
        for name, param in backbone.named_parameters():
            param.required_grad = True

    def __init__(self, cfg):
        """
        """
        super(ModelBase, self).__init__()
        self.logits_type = cfg.MODEL.LOGITS_TYPE # will effect the constructure of net
        self.model_type = cfg.MODEL.MODEL_TYPE
        print ("ModelBase logits type is %s, model type is %s" % (self.logits_type, self.model_type))
        self.resnet = self._get_backbone(self.logits_type, self.model_type) 
        self.emb = torch.nn.Embedding(10, cfg.MODEL.DIM)
        self.mlp = get_mlp(cfg, self.logits_type, self.model_type)
        self.log_sigmoid = nn.LogSigmoid() 
        self.log_softmax = nn.LogSoftmax(dim=-1) 
        self.sigmoid = nn.Sigmoid() 
        self.softmax = nn.Softmax(dim=-1) 
        self.temperature = 0.1
        self.loss_name = "ModelBase"
        self.loss_crit = nn.BCELoss()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def loss(self, logits, batched_inputs):
        """ 
            Hookable 
        """
        raise RuntimeError("Not Implemented ModelBase.loss function")

    def cal_logits(self, images, labels, batched_inputs):
        """ Hookable
        """
        n_batch = labels.shape[0]
        if self.logits_type == 'LABEL_ONLY':
            x = self.resnet.backbone(images).reshape([n_batch, -1])
            x = self.resnet.mlp(x).reshape([n_batch, 10]) # label distribution
            x = self.softmax(x)
            all_label_emb = self.emb(torch.as_tensor(range(0, 10)).long().cuda()) # 10, EMB
            emb1 = (x.reshape([n_batch, 10, 1]) * all_label_emb.reshape([1, 10, -1])).sum(dim=-2) # B, EMB
            logits = self.mlp(torch.cat([emb1, self.emb(labels)], dim=-1))  

        elif self.logits_type == 'DEFAULT':
            if self.model_type == 'resnet18':
                x = self.resnet(images).reshape([n_batch, -1]) # (N, 512)
            else:
                x = self.resnet.feature(images).reshape([n_batch, -1])
 
            x = torch.cat([self.emb(labels), x], dim=-1)   # (N, 512 + 32)
            logits = self.mlp(x)                           # (N, 1)
        return logits

    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        self.output = {}
        b = batched_inputs[0]
        self.resnet.eval()
        n_batch = len(batched_inputs)
        images = stack_data_from_batch(batched_inputs, 'image', torch.float32)
        labels = stack_data_from_batch(batched_inputs, 'category_id', torch.long).reshape([-1]) # (N, )
        logits = self.cal_logits(images, labels, batched_inputs)
        if self.training:
            loss = self.loss(logits, batched_inputs)
            return {
                self.loss_name: loss.mean(),
                **self.output
            }
        else : 
            return self.default_eval(logits)

    def default_eval(self, logits):
        scores = self.sigmoid(logits)
        return scores.detach().cpu().reshape([-1, 1])

class Generator(ModelBase):
    def __init__(self, cfg):
        """ Generator
        """
        super(Generator, self).__init__(cfg)
        self.loss_name = "generator loss"
        self.temperature = 0.3

    def loss(self, logits, batched_inputs):
        logits_copy = logits.reshape([-1]) / self.temperature
        assert 'dis_gt' in batched_inputs[0], "Use Discrimination to generate gt for generator"
        dis_gt = stack_data_from_batch(batched_inputs, 'dis_gt', torch.float32).reshape([-1])
        reward = - torch.log(1 - dis_gt).reshape([-1]) # monotonic increasing
        prob_x = self.softmax(logits_copy.reshape([-1]))
        mean_reward = prob_x * reward
        tmp = (logits.reshape([-1]).topk(10)[0].float()).max()
        self.output['logits max'] = tmp.detach().cpu()
        reward = reward - mean_reward
        loss = - self.log_softmax(logits_copy.reshape([-1]))
        self.output['probs max'] = prob_x.max().detach().cpu()
        self.output['avg reward'] = reward.mean().detach().cpu()
        #proxy_loss = (prob_x * reward * loss).sum()
        proxy_loss = (reward * loss).mean()
        return proxy_loss

    def default_eval(self, logits):
        return logits.detach().cpu().reshape([-1, 1])

class Discriminator(ModelBase):
    def __init__(self, cfg):
        """ Generator
        """
        super(Discriminator, self).__init__(cfg)
        self.loss_name = "discri loss"

    def loss(self, logits, batched_inputs):
        n_batch = len(batched_inputs)
        assert 'is_clean' in batched_inputs[0], "please insert the ground truth from the generator"
        is_clean = stack_data_from_batch(batched_inputs, 'is_clean', torch.float32).reshape([-1])
        is_clean_bool = is_clean.bool()
        is_clean = is_clean.float()
        logits = logits.reshape([-1])  #(N, )
        probs = self.sigmoid(logits)
        self.output['Clean Losses'] = self.loss_crit(probs[0:n_batch], is_clean[0:n_batch]).detach().cpu()
        self.output['Noise Losses'] = self.loss_crit(probs[n_batch:], is_clean[n_batch:]).detach().cpu()
        self.output['accuracy'] = ((probs < 0.5) ^ (is_clean_bool)).float().mean().detach().cpu()
        loss = self.loss_crit(probs, is_clean)
        return loss
