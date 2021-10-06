import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        DIM=32, 
        #LOGITS_TYPE= "LABEL_ONLY"
        LOGITS_TYPE= "DEFAULT",
        MODEL_TYPE= "resnet18",
    ),
    DATASETS=dict(
        CUSTOM_TYPE=("RepeatDataset", dict(times=20)), 
        TRAIN=("cifar_cifar-10_noise-train", ),
        TEST=("cifar_cifar-10_test", ),
        VALIDATION_NUM=100, 
        NOISE_RATIO=0.4, 
        CLEAN_NUM=1000, 
        WITH_BACKGROUND=False, 
    ),
    DATALOADER=dict(NUM_WORKERS=12, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(50, 80),
            MAX_EPOCH=2,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            BASE_LR=0.001,
            MOMENTUM=0.95,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.3,
        ),
        CHECKPOINT_PERIOD=20,
        IMS_PER_BATCH=64,
        IMS_PER_DEVICE=64,
    ),
    TEST=dict(
        #EVAL_PERIOD=1,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
            ],
            TEST_PIPELINES= [
            ], 
        )
    ),
    OUTPUT_DIR=osp.join(
        '/home/xiongkun/Output',
        'gan-reweight',
    ),
    SEED=16925062,
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
