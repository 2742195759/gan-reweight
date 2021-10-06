import torch
from visdom import Visdom
import numpy as np
import time

vis = Visdom('108.160.136.227', port=23333, env='tmp')

def draw_histogram(tensor, bincount=30, prefix=''):
    vis.histogram(
        X=tns.reshape([-1]),
        opts={'xlabel': 'bin', 'ylabel': 'count', 'title': prefix + 'histogram', 'numbins': bincount}
    )

tns = torch.normal(0, 1.0, (10, 100))
draw_histogram(tns, 30, 'sdf')
