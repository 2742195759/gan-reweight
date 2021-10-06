import logging
import torch
import torch.nn as nn
from cvpods.modeling.xkmodel import MLP, SoftmaxPredictor
from cvpods.utils.xklib import stack_data_from_batch
import torchvision.models as models
import os
parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent)
from networks import *

def build_model(cfg):
    depth = cfg.MODEL.DEPTH
    width = cfg.MODEL.WIDTH
    drop_out = cfg.MODEL.DROP_OUT
    class_num = cfg.MODEL.CLASS
    if cfg.MODEL.NET_TYPE == 'wide-resnet': 
        model = wide_resnet.Wide_ResNet(depth, width, drop_out, class_num)

    logger = logging.getLogger(__name__)
    logger.info("Backbone Model:\n{}".format(model))
    return model
