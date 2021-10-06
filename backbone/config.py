import os.path as osp
import torchvision.transforms as transforms
from cvpods.configs.base_classification_config import BaseClassificationConfig
"""
backbone from https://github.com/meliketoy/wide-resnet.pytorch
contains 
    wide_resnet: 28 x 10 
    resnet
    vgg 

and the pretrained model can achieve 95% ACC_top1 in cifar-10 dataset
"""
_config_dict = dict(
    MODEL=dict(
        NET_TYPE='wide-resnet', 
        DEPTH=28, 
        WIDTH=10, 
        WEIGHTS="",
        DROP_OUT=0.3,
        CLASS=10,
    ),
    SEED=16925062,
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
