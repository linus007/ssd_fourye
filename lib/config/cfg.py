import numpy as np
"""
config for datasets
"""
DATA_ROOT = "../data"

PASCAL_VOC2007 = "pascal_voc2007"


"""
config for the common
"""
DATA_SET_USED = PASCAL_VOC2007
ESP = 1e-14
IMAGE_SIZE = 300
BATCH_SIZE = 16
CACHE_SIZE = 10


# feature map size used to detect the object in conv layers
FP_SIZE = [38, 19, 10, 5, 3, 1]
NUM_DEFAULT_BOXES = [4, 6, 6, 6, 4, 4]
assert len(FP_SIZE) == len(NUM_DEFAULT_BOXES)
NUM_FEATUE_MAPS_USED = len(FP_SIZE)



M = NUM_FEATUE_MAPS_USED
S_MIN = 0.2
S_MAX = 0.9
#S = [S_MIN + (S_MAX - S_MIN) * ((k - 1) / (M - 1)) for k in range(1, M + 1)]
S = [0.1, 0.34, 0.48, 0.62, 0.76, 0.9, 1.0]
RAS_NORMAL = [1.0, 0.5, 2.0, 3.0, 1.0 / 3.0]
S_RATIO_OF_1 = [0.1, np.sqrt(S[1] * S[2])
    , np.sqrt(S[2]*S[3]), np.sqrt(S[3]*S[4])
    , np.sqrt(S[4]*S[5]), np.sqrt(S[5]*S[6])]


DG_THRESHOLD = 0.5
BN_EPSILON = 0.001


"""
Solver configure
"""
SOLVER_NAME = "solver.vgg16_solver.VGG16_Solver"

IS_PRETRAIN = True
PRETRAIN_MODEL_PATH = "pretrain/save.ckpt-0"
TRAIN_DIR = "train"

MOMENT = 0.9
GPU = 0

LEARNING_RATE = 0.0001
DECAY_STEPS = 20000
DECAY_RATE = 0.1
STAIR_CASE = True

MAX_ITERS = 60000
SAVE_ITER = 1000
SUMMARY_ITER = 10

THRESHOULD = 0.2
IOU_THRESHOULD = 0.5

"""
Common configure
"""
