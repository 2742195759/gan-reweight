import logging#{{{
import os
import pickle as pkl
import sys
sys.path.insert(0, '.')  # noqa: E402
from collections import OrderedDict
import torch
import tqdm
from colorama import Fore, Style
import cvpods.data.transforms as T
from cvpods.engine.base_runner import RunnerBase
from cvpods.checkpoint import Checkpointer
from cvpods.data import build_test_loader, build_train_loader
from cvpods.evaluation import (
    DatasetEvaluator, inference_on_dataset,
    print_csv_format, verify_results
)
from cvpods.modeling.nn_utils.precise_bn import get_bn_modules
from cvpods.solver import build_lr_scheduler, build_optimizer
from cvpods.utils import (
    CommonMetricPrinter, JSONWriter, PathManager,
    TensorboardXWriter, collect_env_info, comm,
    seed_all_rng, setup_logger, VisdomWriter
)
from cvpods.engine import DefaultRunner, default_argument_parser, default_setup, hooks, launch
from cvpods.evaluation import ClassificationEvaluator
from cvpods.utils import comm
from cvpods.utils import EventStorage
import cvpods.model_zoo as model_zoo
from config import clean_cfg, noise_cfg, global_cfg
from sklearn.metrics import roc_auc_score
import copy
import sys
import numpy as np
import show_distribution

if global_cfg.GAN.BACKBONE_TYPE == 'resnet18':
    from baseline.config_for_gan import config as baseline_cfg
elif global_cfg.GAN.BACKBONE_TYPE == 'widenet': 
    from widenet.config_for_gan import config as baseline_cfg
    
#}}}

if True:  # "INIT"/*{{{*/
    backbone_type = global_cfg.GAN.BACKBONE_TYPE
    if backbone_type == 'widenet':
        backbone_path = "widenet"
    elif backbone_type == 'resnet18':
        backbone_path = "resnet18"
        
    baseline = model_zoo.get(
        backbone_path,
        playground_path="/home/data/GAN/gan-based",
        custom_config=dict(
            MODEL=dict(
                WEIGHTS=("" if not global_cfg.GAN.RESUME 
                    else "/home/data/Output/resnet18-cifar/model_final.pth"),
                SAMPLE_WEIGHT="",
            )
        )
    )
    gen = model_zoo.get(
        "gen_net",
        playground_path="/home/data/GAN/gan-based",
        custom_config={
            "MODEL": {
                "MODEL_TYPE": backbone_type,
            }
        }
    )
    dis = model_zoo.get(
        "dis_net",
        playground_path="/home/data/GAN/gan-based",
        custom_config={
            "MODEL": {
                "MODEL_TYPE": backbone_type,
            }
        }
    )
    """ for reuse, we set the 
    """
    clean_dataloader = build_train_loader(clean_cfg)
    noise_dataloader = build_train_loader(noise_cfg)#/*}}}*/

class Cvpack2DataloaderWrapper: #/*{{{*/
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self.sampler = dataloader.sampler

    def __iter__(self):
        self.next_batch = []
        self.next_idx   = 0
        self.iter = iter(self.dataloader)
        return self

    def __next__(self):
        while (self.next_idx + 1 > len(self.next_batch)):
            data = next(self.iter)
            self.next_batch = self.process(data)
            self.next_idx = 0
        new_data = self.next_batch[self.next_idx]
        self.next_idx += 1
        return new_data
    
    def __len__(self):
        return len(self.dataloader)

    def process(self, data):
        raise NotImplementedError("Please implement the process function.")

class GenDataloaderWrapper(Cvpack2DataloaderWrapper):
    def __init__(self, dataloader, dis_model):
        Cvpack2DataloaderWrapper.__init__(self, dataloader)
        self.dis_model = dis_model

    def process(self, data) : 
        self.dis_model.eval()
        scores = self.dis_model(data).reshape([-1]).tolist()
        for i, s in enumerate(scores):
            data[i]['dis_gt'] = s
        return [data]

class DisDataloaderWrapper(Cvpack2DataloaderWrapper):
    def __init__(self, dataloader):
        Cvpack2DataloaderWrapper.__init__(self, dataloader)

    def process(self, data) : 
        ret = []
        batch_size = len(data)
        #assert batch_size % 2 == 0, "Use DisDataloaderWrapper must ensure batch_size is even"
        i = 0
        ret.append(
            data[i*batch_size:(i+1)*batch_size] 
            + self.fake[i*batch_size:(i+1)*batch_size])
        assert (ret[0].__len__() == 2 * batch_size)
        return ret

    def __len__(self):
        return len(self.dataloader)

    def insert_fake(self, fake):
        self.fake = fake
        self.noise_size = len(self.fake)
#}}}

class GenTrainer(DefaultRunner): #{{{
    def __init__(self, cfg, model, is_train_backbone):
        super().__init__(cfg, lambda x: model)
        self.is_train_backbone = is_train_backbone

    @classmethod
    def build_train_loader(cls, cfg):
        data_loader = build_train_loader(cfg)
        return GenDataloaderWrapper(data_loader, dis)

    def train(self):
        self.model.set_backbone(copy.deepcopy(baseline))
        super(GenTrainer, self).train()
        self.model.reset_backbone(copy.deepcopy(baseline))
#}}}

class DisTrainer(DefaultRunner): #{{{
    def __init__(self, cfg, model, is_train_backbone):
        super().__init__(cfg, lambda x: model)
        self.is_train_backbone = is_train_backbone

    def train(self):
        self.model.set_backbone(copy.deepcopy(baseline))
        super(DisTrainer, self).train()
        self.model.reset_backbone(copy.deepcopy(baseline))

    @classmethod
    def build_train_loader(cls, cfg):
        data_loader = build_train_loader(cfg)
        return DisDataloaderWrapper(data_loader)
#}}}

class BaselineTrainer(DefaultRunner): #{{{
    def print_distrib(self, sample_number=20):
        outputs = []
        sample_num = sample_number
        for batch in self.data_loader:
            sample_num -= 1
            if sample_num == 0 : break ;
            self.model.eval()
            outputs.append( self.model(batch) )
        outputs = torch.cat(outputs, dim=0)
        assert outputs.shape[-1] == 10, "Must be ad 0-9 distribution"
        for i in range(10):
            self.father_storage.put_scalar(
                "backbone-var-" + str(i), 
                outputs[:, i].std(),
            )
        
    def __init__(self, cfg, model, father_storage=None):
        super(BaselineTrainer, self).__init__(cfg, lambda x: model)
        self.father_storage = father_storage

    def train(self, is_init=False):
        if not is_init:
            print ("Loading sample weights")
            baseline.load_weight(global_cfg.GAN.SAMPLE_WEIGHT_PATH)
        else : 
            assert not hasattr(self, 'image2weight'), "When first train, the image2weight must be None"
        super(BaselineTrainer, self).train()
        #self.print_distrib()
        """load best weights, aimed to increase the AUC"""
        self.checkpointer.load(global_cfg.GAN.BACKBONE_BEST_PATH)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, meta, output_folder=None):
        evaluator_type = meta.evaluator_type
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if evaluator_type == "classification":
            return ClassificationEvaluator(dataset_name, meta, cfg, True, output_folder, True)
#}}}

all_sampled = set()

class GANTrainer(RunnerBase):  #{{{
    def __init__(self, cfg): #{{{
        super(GANTrainer, self).__init__()
        self.step = 0
        self.setup(cfg)
        self.register_hooks(self.build_hooks(cfg))#}}}
    def build_hooks(self, cfg):#{{{
        ret = []
        if hasattr(cfg, 'VISDOM') and cfg.VISDOM.TURN_ON == True:
            logger = logging.getLogger("Visdom")
            logger.info('Host:' + str(cfg.VISDOM.HOST), 'Port:' + str(cfg.VISDOM.PORT))
            ret.append(hooks.PeriodicWriter([VisdomWriter(cfg.VISDOM.HOST, cfg.VISDOM.PORT, 1, cfg.VISDOM.KEY_LIST, cfg.VISDOM.ENV_PREFIX)], 1))
        return ret
#}}}
    def cal_reweight(self):#{{{
        show_distribution.main(gen)#}}}
    def _gen_fake_data_by_method(self, method, logits, size):#{{{
        if method == 'sample':
            return self._gen_fake_data_sample(size, logits)
        elif method == 'topk': 
            return self._gen_fake_data_topk(size, logits)#}}}
    def _gen_fake_data_sample(self, sample_size, logits):#{{{
        """ return idxs: list like
            param: logits shape = N, 
        """
        logits = logits.reshape([-1])
        length= logits.shape[0]
        prob = torch.functional.F.softmax(logits, dim=-1)
        indices = np.random.choice(length, (sample_size, ), replace=True, p=prob.numpy())
        print ("Sample Number:", len(set(list(indices))))
        return indices#}}}
    def _gen_fake_data_topk(self, sample_size, logits):#{{{
        """ return idxs: list(int) like
            param: logits shape = N, 
        """
        indices = (torch.topk(logits[:,0], sample_size)[1]).tolist()
        return indices#}}}
    def _gen_fake_data(self, method, size=None):#{{{
        """ check if the selected sampler is 
        """
        output = []
        labels = []
        data_tmp = []
        clean_size = size
        if size == None:
            clean_size = noise_cfg.DATASETS.CLEAN_NUM

        print ( "Inserting fake samples" )
        gen.eval()
        for data in tqdm.tqdm(noise_dataloader): 
            output.append(gen(data).reshape([-1,1]))
            labels.extend([i['is_clean'] for i in data])
            data_tmp.extend(data)
        output = torch.cat(output, dim=0)

        indices = self._gen_fake_data_by_method(method, output, clean_size)
        data_fake = []
        choise_clean_rate = 0.0
        for idx in indices: 
            choise_clean_rate += labels[idx]
            data_tmp[idx]['is_clean'] = 0  # 0 is noise, 1 is clean
            data_fake.append(data_tmp[idx])
            all_sampled.add(data_tmp[idx]['image_id'])

        print ("Tot Neg : ", len(all_sampled))
        auc = roc_auc_score(np.array(labels), output.numpy())
        self.storage.put_scalar("generator right rate", choise_clean_rate / len(indices) * 1.0)
        self.storage.put_scalar("generator auc", auc)
        
        print ("[Generator Right Rate]:", choise_clean_rate / len(indices) * 1.0)
        print ("[Generator AUC]:", auc)
        return data_fake#}}}
    def setup(self, cfg): #{{{
        cfg.link_log()
        self.start_iter = cfg.GAN.START_ITER
        self.max_iter   = cfg.GAN.MAX_ITER
        self.epoch_dis = cfg.GAN.EPOCH_DIS
        self.epoch_gen = cfg.GAN.EPOCH_GEN 
        self.interval = cfg.GAN.HISTOGRAM_INTERVAL 
        self.backbone_interval = cfg.GAN.TRAIN_BACKBONE_INTERVAL
        self.is_train_backbone = cfg.GAN.TRAIN_BACKBONE
        if cfg.GAN.RESUME:
            print ("Resuming")
            self.resume = True
        else:
            print ("From Stratch")
            self.resume = False
        if self.resume: self.resume_model()
#}}}
    def resume_model(self):#{{{
        self.dis_trainer = DisTrainer(copy.deepcopy(clean_cfg), dis, self.is_train_backbone)
        self.gen_trainer = GenTrainer(copy.deepcopy(noise_cfg), gen, self.is_train_backbone)
        self.dis_trainer.resume_or_load(False)  # just load the weigths from MODEL.WEIGHTS""
        self.gen_trainer.resume_or_load(False)#}}}
    def eval_dis_model(self, dataloader, prefix='noise', mode='eval'):#{{{
        if mode == 'eval':
            dis.eval()
        else : 
            dis.train()
        hit = 0.0
        tot = 0.0
        clean_tot = 0
        losses = 0.0
        neg_nr = 0.0
        outputs = []
        for batch in tqdm.tqdm(dataloader): 
            output = dis(batch).reshape([-1]).cpu().detach()
            neg_nr += (output < 0.5).sum()
            outputs.append(output)
            label  = torch.as_tensor([ b['is_clean'] for b in batch ]).float()
            clean_tot += label.sum().cpu()
            losses += torch.nn.BCELoss(reduction='sum')(output, label)
            for idx, item in enumerate(output.tolist()): 
                if item > 0.5 and batch[idx]['is_clean'] == 1: hit += 1
                if item <=0.5 and batch[idx]['is_clean'] == 0: hit += 1
            tot += len(batch)
        outputs = torch.cat(outputs[:5], dim=-1)
        print ("NegNumber    : ", neg_nr)
        print ("Discriminator: ", hit / tot)
        print ("Clean Rate   : ", clean_tot / tot)
        print ("Losses       : ", losses / tot)
        if prefix : 
            self.storage.put_scalar("dis-" + prefix + '-loss', losses / tot)
            if self.step % self.interval == 0:
                self.storage.put_tensor(outputs, 'histogram', {
                        'title': 'dis-'+prefix+'-'+str(self.step), 
                        'bincount': 200,
                    })
#}}}
    def run_step(self):#{{{
        """
        Run just a iteration inside the EventStorage
        """
        self.step += 1
        print ("#" * 80)
        print ("Current Step:", self.step)

        if self.is_train_backbone and (( self.step - 1 ) % self.backbone_interval == 0): 
            for _ in range(1):
                print ("######### Train Backbone model")
                self.cal_reweight() # calculate the weight of samples
                baseline_trainer = BaselineTrainer(copy.deepcopy(baseline_cfg), baseline, self.storage)
                baseline.train()
                baseline_trainer.train(self.step == 1)
                results = baseline_trainer._max_eval_results
                self.storage.put_scalar('backbone-Acc',
                    results['Accuracy']['Top_1 Acc'])

        print ("######### Train Discriminative model")
        fake_data = self._gen_fake_data(global_cfg.GAN.FAKE_DATA_METHOD)
        for _ in range(self.epoch_dis): 
            self.dis_trainer = DisTrainer(copy.deepcopy(clean_cfg), dis, self.is_train_backbone)
            self.dis_trainer.data_loader.insert_fake(fake_data)
            dis.train()
            self.dis_trainer.train()  # train dis model with clean data and fake data from generative model

        #print ("Eval of Discriminator")#{{{
        #self.eval_dis_model(
        #    noise_dataloader, 'noise', 'eval'
        #)
        #self.eval_dis_model(
        #    clean_dataloader, 'clean', 'eval'
        #)
#}}}
        # Update the gen model
        print ("######### Train Generative model")
        for _ in range(self.epoch_gen): 
            self.gen_trainer = GenTrainer(copy.deepcopy(noise_cfg), gen, self.is_train_backbone)
            gen.train()
            self.gen_trainer.train()
#}}}
    def start(self):#{{{
        self.train(self.start_iter, self.max_iter, 1)
        #self.baseline_task.train()#}}}
#}}}

# main function{{{
import matplotlib.pyplot as plt
import visdom
import time

def draw_multi_histogram(tensor_list, bincount=70, prefix=''):
    fig,ax=plt.subplots(figsize=(8,5))
    array_list = [ tensor.numpy() for tensor in tensor_list ]
    plt.xlabel('logits before softmax')
    plt.ylabel('frequence')
    ax.set_title(prefix + '-histogram')
    ax.hist(array_list, bincount, histtype='bar',label=["clean", "flip"])
    ax.legend()
    plt.savefig('matplotlib-output.jpg')

def save_weight_logits(image_ids, outputs, categorys):
    to_save= {}
    verify = {}
    assert len(image_ids) == len(outputs)
    assert isinstance(image_ids, np.ndarray)
    assert isinstance(outputs, np.ndarray)
    assert len(image_ids.shape) == 1
    assert len(outputs.shape) == 1
    for id, output, cate in zip(image_ids, outputs, categorys):
        to_save[id] = output
        verify[id] = cate
    import pickle
    pickle.dump({'weights': to_save, 'verify': verify}, open("./cache/weights.pkl", "wb"))

from cvpods.utils import PltHistogram

def main(generator):
    with EventStorage() as storage:
        visdom_writer = VisdomWriter('192.168.1.1', '8082', 1, [], 'test') 
        hist = PltHistogram()
        output = []
        labels = []
        data_tmp = []
        image_ids = []
        temp = 0.8
        clean_size = noise_cfg.DATASETS.CLEAN_NUM
        generator.eval()
        categorys = []
        for data in tqdm.tqdm(noise_dataloader): 
            for d in data : 
                if d['image_id'] == 0: print (d)
            output.append(generator(data).reshape([-1,1]))
            labels.extend([i['is_clean'] for i in data])
            image_ids.extend([i['image_id'] for i in data])
            categorys.extend([i['category_id'] for i in data])
            data_tmp.extend(data)
        output = torch.cat(output, dim=0)
        auc = roc_auc_score(np.array(labels), output.numpy())
        print ("AUC:", auc)
        method = "sigmoid"
        output = output.reshape([-1])
        labels = torch.as_tensor(labels)
        image_ids = torch.as_tensor(image_ids)

        if method == 'sigmoid':
            output = (output - output.mean()) / output.std()
            output = torch.nn.Sigmoid()(output)

        if method == 'softmax':
            output = torch.nn.Softmax(dim=0)(output / temp)
            output = (output) / output.mean()

        if method == 'upbound':
            """ classify by ratio of 
            """
            bin_count = 70
            minn = output.min().item()
            maxx = output.max().item()

            def get_freq(input, bin_count, minn, maxx):
                output_arr = input.numpy()
                hist, _ = np.histogram(output_arr, bins=bin_count, range=(minn, maxx))
                return hist

            hist_clean = get_freq(output[labels==1], 70, minn, maxx)
            hist_noise = get_freq(output[labels==0], 70, minn, maxx)
            output_arr = output.numpy()
            interval = (maxx - minn) / (bin_count - 1)
            output_int = ((output_arr - minn) / interval).astype('int')
            output_weight = hist_clean[output_int] / (hist_noise[output_int] + hist_clean[output_int])
            output = torch.as_tensor(output_weight)

        if method == 'clip':
            ratio = 0.8
            length = len(output)
            top_k = int(length * ratio)
            indices = torch.topk(output.reshape([-1]), top_k)[1]
            labels_tmp = labels[indices]
            output = torch.zeros_like(output)
            output[indices] = 1.0
            clean_ratio = labels_tmp.sum()*1.0 / len(labels_tmp)
            print ("Ratio", clean_ratio)
            print ("Clean / Tot", clean_ratio * ratio)

        if method == 'best':
            clean_weight = 0.9
            noise_weight = 0.1
            output[labels==1] = 0.9
            output[labels==0] = 0.1

        image = hist([output[labels==1], output[labels==0]], 
            70, prefix="sigmoid", xylabels=['logits', 'frequence'], label=['clean', 'noise'])
        storage.put_image("histogram", image)
        save_weight_logits(image_ids.numpy(), output.numpy(), categorys)
        visdom_writer.write()

if __name__ == "__main__":
    main(gen)
#}}}
