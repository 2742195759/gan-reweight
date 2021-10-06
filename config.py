import os.path as osp
import torchvision.transforms as transforms
    
from cvpods.configs.base_classification_config import BaseClassificationConfig

""" A Dataset is effected by the following hyperparameters
    EPOCH / BATCH_SIZE / DATASETS
"""

_config_dict = dict(
    MODEL=dict(
        WEIGHTS="/home/data/Output/dis_model/model_final.pth",
    ),
    DATASETS=dict(
        CUSTOM_TYPE=('RepeatDataset', dict(times=1)),
        ROOT="/home/data/dataset",
        TRAIN=("cifar_cifar-10_clean", ),
        VALIDATION_NUM=0, 
        NOISE_RATIO=0.4, 
        CLEAN_NUM=1000, 
        WITH_BACKGROUND=False, 
    ),
    DATALOADER=dict(
        NUM_WORKERS=0, 
        SAMPLER_TRAIN="DistributedGroupSamplerTimeSeed",
    ),
    SOLVER=dict(
        LR_SCHEDULER=dict(
            MAX_EPOCH=30,
            #MAX_ITER=20,
            WARMUP_ITERS=0,
        ),
        IMS_PER_BATCH=128,
        IMS_PER_DEVICE=128,
        OPTIMIZER=dict(
            BASE_LR=0.01,
            MOMENTUM=0.90,
            WEIGHT_DECAY=1e-4,
            WEIGHT_DECAY_NORM=1e-4,
            GAMMA=0.3,
        ),
    ),
    SEED=16925062,
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
            ],
            TEST_PIPELINES= [
            ], 
        )
    ),
    OUTPUT_DIR=osp.join(
        '/home/data/Output',
        'dis_model',
    ),
)

class CleanConfig(BaseClassificationConfig):
    def __init__(self):
        super(CleanConfig, self).__init__()
        self._register_configuration(_config_dict)

class NoiseConfig(CleanConfig):
    def __init__(self):
        super(NoiseConfig, self).__init__()
        self._register_configuration(dict(
            MODEL=dict(
                WEIGHTS="/home/data/Output/gen_model/model_final.pth"
            ),
            DATASETS=dict(
                TRAIN=("cifar_cifar-10_noise-train", ),
                CUSTOM_TYPE=('ConcatDataset', dict()),
            ), 
            SOLVER=dict(
                LR_SCHEDULER=dict(
                    MAX_EPOCH=2,
                    #MAX_ITER=20,
                ),  
                OPTIMIZER=dict(
                    BASE_LR=0.01,
                    MOMENTUM=0.90,
                    WEIGHT_DECAY=2e-4,
                    WEIGHT_DECAY_NORM=1e-4,
                    GAMMA=0.3,
                ),
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'gen_model',
            ),
        ))


#backbone_var_list = [ 'backbone-var-' + str(i) for i in range(10) ]
backbone_var_list = []
class GANConfig(BaseClassificationConfig):
    def __init__(self):
        super(GANConfig, self).__init__()
        self._register_configuration(dict(
            GAN=dict(
                MAX_ITER=5000, # max iteration time
                START_ITER=0, # set it to zero
                EPOCH_DIS=1 , # the training epoch of Discriminator during a iteration
                EPOCH_GEN=1 , # the training epoch of Generator during a iteration
                RESUME=False, # load the generator and discriminator
                HISTOGRAM_INTERVAL=50000, # interval for sending histogram, but will cost a mount of time
                TRAIN_BACKBONE=True, # False for fixed backbone, True for 3 model iterative training 
                TRAIN_BACKBONE_INTERVAL=30, #after this iteration, we retrain the backbone using updated generator 
                FAKE_DATA_METHOD='sample',  # 'sample' | 'topk'

                SAMPLE_WEIGHT_PATH='/home/data/GAN/gan-based/cache/weights.pkl',  

                BACKBONE_TYPE ='widenet', # baseline type: resnet18 / widenet{{{
                BACKBONE_BEST_PATH='/home/data/Output/widenet28-cifar/model_best.pth',#}}}

                #BACKBONE_TYPE ='resnet18', # baseline type: resnet18 / widenet{{{
                #BACKBONE_BEST_PATH='/home/data/Output/resnet18-cifar/model_best.pth',#}}}
            ),
            OUTPUT_DIR=osp.join(
                '/home/data/Output',
                'gan-model',
            ),
            VISDOM=dict(
                HOST="192.168.1.1", 
                PORT="8082", 
                TURN_ON=True,
                ENV_PREFIX='gan-9-13-',
                KEY_LIST=['dis-noise-loss', 
                          'dis-clean-loss', 
                          'generator right rate', 
                          'generator auc', 
                          'backbone-Acc',
                          ] + backbone_var_list
            ), 
        ))

clean_cfg = CleanConfig()
noise_cfg = NoiseConfig()
global_cfg = GANConfig()

#model = "debug"
#model = "reproduct"
model = "train"

if model == 'debug': 
    print ("Debug Mode")
    global_cfg.GAN.RESUME=False
    global_cfg.GAN.TRAIN_BACKBONE=False
    global_cfg.VISDOM.TURN_ON=False
elif model == 'reproduct': 
    print ("Reproduct Mode")
    global_cfg.GAN.RESUME=True
    global_cfg.GAN.TRAIN_BACKBONE=False
    global_cfg.VISDOM.TURN_ON=False
else:
    print ("Train Mode")
    pass
