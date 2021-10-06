import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="",
        SAMPLE_WEIGHT="", 
    ),
    DATASETS=dict(
        CUSTOM_TYPE=("RepeatDataset", dict(times=10)), 
        ROOT="/home/data/dataset/",
        TRAIN=("cifar_cifar-10_noise-train", ),
        TEST=("cifar_cifar-10_test", ),
        VALIDATION_NUM=0, 
        NOISE_RATIO=0.4, 
        DISCOUNT_RATIO=1.0, # reduce the training sample : len = len * discount
        CLEAN_NUM=0, 
        WITH_BACKGROUND=True, 
    ),
    DATALOADER=dict(NUM_WORKERS=12, ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            STEPS=(6, 12, 16),
            MAX_EPOCH=20,
            WARMUP_ITERS=0,
        ),
        OPTIMIZER=dict(
            #BASE_LR=0.1 / 4.0,
            BASE_LR=0.1,
            MOMENTUM=0.90,
            WEIGHT_DECAY=5e-4,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.2,
        ),
        CHECKPOINT_PERIOD=100,
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=128,
    ),
    TEST=dict(
        EVAL_PERIOD=1,
    ),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                ("Torch_RC", transforms.RandomCrop(32, padding=4)),
                ("Torch_RHF", transforms.RandomHorizontalFlip())
            ],
            TEST_PIPELINES= [
            ], 
        )
    ),
    OUTPUT_DIR=osp.join(
        '/home/data/Output',
        'baseline-widenet',
    ),
    SEED=16925062,
    VISDOM=dict(
        HOST="192.168.1.1", 
        PORT="8082", 
        TURN_ON=True,
        ENV_PREFIX='baseline-widenet',
        KEY_LIST=['Accuracy/Top_1 Acc']  # bacause flattened
    ), 
)


class CustomerConfig(BaseClassificationConfig):
    def __init__(self):
        super(CustomerConfig, self).__init__()
        self._register_configuration(_config_dict)

config = CustomerConfig()
