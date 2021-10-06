# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
import torch.nn as nn
from cvpods.modeling.xkmodel import MLP, SoftmaxPredictor
from cvpods.utils.xklib import stack_data_from_batch
import torchvision.models as models
from cvpods import model_zoo
import sys
import os
current_path = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(current_path + '../'))
import common

def build_model(cfg):
    model = common.Discriminator(cfg)
    logger = logging.getLogger(__name__)
    logger.info("Discrimination Model is:\n{}".format(model))
    return model
