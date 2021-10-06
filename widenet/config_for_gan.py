import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        SAMPLE_WEIGHT=""
    ),
    DATASETS=dict(
        CUSTOM_TYPE=("RepeatDataset", dict(times=5)), 
        ROOT="/home/data/dataset",
        TRAIN=("cifar_cifar-10_noise-train", ),
        TEST=("cifar_cifar-10_test", ),
        VALIDATION_NUM=0, 
        NOISE_RATIO=0.4, 
        CLEAN_NUM=1000, 
        WITH_BACKGROUND=False, 
    ),
    DATALOADER=dict(
        NUM_WORKERS=12,
        SAMPLER_TRAIN="DistributedGroupSamplerTimeSeed",
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(3, 6, 8),
            #MAX_ITER=15000,
            MAX_EPOCH=10,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.1 ,
            MOMENTUM=0.90,
            WEIGHT_DECAY=5e-4,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.2,
        ),
        CHECKPOINT_PERIOD=2,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=128,
    ),
    TEST=dict(
        EVAL_PERIOD=1,
        SORT_BY='Accuracy/Top_1 Acc',
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RC", transforms.RandomCrop(32, padding=4)),
                ("Torch_RHF", transforms.RandomHorizontalFlip()),
            ],
            TEST_PIPELINES= [
            ], 
        ),
    ),
    OUTPUT_DIR=osp.join(
        '/home/data/Output',
        'widenet28-cifar',
    ),
    SEED=16925062,
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
