import logging
import torch
import torch.nn as nn
from cvpods.modeling.xkmodel import MLP, SoftmaxPredictor
from cvpods.utils.xklib import stack_data_from_batch
import torchvision.models as models
from cvpods import model_zoo
import sys
sys.path.insert(0, '../')

class Bpr(nn.Module):
    """
        the simplest recommendation model, used to test the successful of 
        transfer cvpods to recommendation area
    """
    def load_weight(self, path):
        import pickle
        logger = logging.getLogger(__name__)
        logger.info("Loading sample weights from " + path)
        with open (path, 'rb') as fp:
            tmp = pickle.load(fp)
            self.image2weight = tmp['weights']
            self.verify = tmp['verify'] # for verify the dataset is the same

    def __init__(self, cfg):
        """
        """
        super(Bpr, self).__init__()
        class_number = 10
        if 'cifar-100' in cfg.DATASETS.TRAIN[0]: 
            class_number = 100
        self.backbone = model_zoo.get(
                "backbone", False, 
                dict(
                    MODEL=dict(
                        #WEIGHTS="/home/data/cvpods_model/reweight/gan-based/baseline3/wide-resnet.pytorch/checkpoint/cifar10/wide-resnet-28x10.pth",
                        WEIGHTS="",
                        CLASS=class_number,
                    ),
                )
                , "/home/data/GAN/gan-based/"
            )
        self.log_softmax = nn.LogSoftmax(dim=-1) 
        self.loss_crit = nn.NLLLoss(reduction='none')
        self.image2weight = None
        self.verify = None
        if cfg.MODEL.SAMPLE_WEIGHT: self.load_weight(cfg.MODEL.SAMPLE_WEIGHT)

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)

    def _verify(self, batched_inputs):
        if not self.verify: return
        for b in batched_inputs:
            #print (self.verify[b['image_id']], b['category_id'])
            assert b['category_id'] == self.verify[b['image_id']], "Not the same dataset, Check the DATASETS config"

    def forward(self, batched_inputs):
        """
        Input:
        Output:
        """
        b = batched_inputs[0]
        n_batch = len(batched_inputs)
        images = stack_data_from_batch(batched_inputs, 'image', torch.float32)
        #images = images.reshape(n_batch, -1)
        labels = stack_data_from_batch(batched_inputs, 'category_id', torch.int64)
        if self.training:
            self._verify(batched_inputs)
            labels = stack_data_from_batch(batched_inputs, 'category_id', torch.int64)  # NLLloss only support labels = [nbatch, ]
            if self.image2weight : 
                weights = torch.as_tensor([self.image2weight.get(b['image_id'], 1.0) for b in batched_inputs]).float().to(self.device)
            else : 
                weights = torch.ones((n_batch, 1)).float().to(self.device)

            weights = weights.reshape([-1, 1])
            logits = self.backbone(images)
            loss = self.loss_crit(self.log_softmax(logits), labels).reshape([-1, 1])
            loss = (loss * weights).mean()
            return {
                "loss_cls": loss,
            }
        else : 
            #assert len(batched_inputs) == 1, "eval mode need batchsize {} == 1".formated(batched_inputs)
            scores = self.backbone(images)
            return scores.detach().reshape([n_batch, -1])

def build_model(cfg):

    model = Bpr(cfg)

    logger = logging.getLogger(__name__)
    logger.info("Model:\n{}".format(model))
    return model
