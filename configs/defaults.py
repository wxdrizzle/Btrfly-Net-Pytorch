from yacs.config import CfgNode as CN

_C = CN()

_C.TRAIN_WEIGHT = "/shenlab/local/zhenghan/pj_wx/BtrflyNet/datasets/Label2D/weight/train_weight.mat"

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.USE_GAN = 0
_C.MODEL.IMAGE_SIZE = 512
_C.MODEL.USE_BN = 1
_C.MODEL.CHANNELS = (1, 32, 64, 128, 256, 256, 512, 1024, 512, 512, 256, 128, 64, 25)

_C.SOLVER = CN()
_C.SOLVER.LR = 1e-3
_C.SOLVER.WEIGHT_DECAY = 1e-3
_C.SOLVER.MAX_ITER = 120000
_C.SOLVER.BATCH_SIZE = 16
_C.SOLVER.SAVE_NUM = 25

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 10
_C.TEST.DEVICE = 'cuda'

_C.ORIGINAL_PATH = '/shenlab/local/zhenghan/original/'
_C.MAT_DIR_TRAIN = "datasets/Label2D/train/"
_C.INPUT_IMG_DIR_TRAIN = "datasets/JPEGUncover_train/"

_C.CROP_INFO_DIR = 'datasets/Crop_info'
_C.MAT_DIR_VAL = 'datasets/VOC2007/Label2D_val/'
_C.INPUT_IMG_DIR_VAL = 'datasets/VOC2007/JPEGUncover_val/'

_C.MAT_DIR_TEST = 'datasets/'
_C.INPUT_IMG_DIR_TEST = 'datasets/'

_C.OUTPUT_DIR = "outputs"

cfg = _C