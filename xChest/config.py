
PATH_BASE = "D:/Download/chest_xray/"
PATH_PLOT_FOLDER = "xChest/plots/"
PATH_MODEL_FOLDER = "xChest/models/"
PATH_SUB = ["train/", "test/", "val/"]

IMG_SIZE = (244,244)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 16

class HyperPars:
    def __init__(self):
        self.BATCH_NORM_AXIS = -1
        self.BATCH_NORM_MOMENTUM = 0.99
        self.BATCH_NORM_EPSILON = 0.001
        self.DENSE_UNITS = 256
        self.KERNEL_REGULARIZER_L2 = 0.016
        self.ACTIVITY_REGULARIZER_L1 = 0.06
        self.BIAS_REGULARIZER_L1 = 0.006
        self.DROPOUT_RATE = 0.04
        self.DROPOUT_SEED = 75
        self.LEARNING_RATE = 0.0001
        self.FUNC_ACTIVATION_RELU = "relu"
        self.FUNC_ACTIVATION_SOFTMAX = "softmax"
        self.FUNC_LOSS = "categorical_crossentropy"
        self.LOSS_METRIC = "accuracy"

    def print_info(self):
        print("[I] Hyperparameters used:")
        print("[I] BATCH_NORM_AXIS:", self.BATCH_NORM_AXIS)
        print("[I] BATCH_NORM_MOMENTUM:", self.BATCH_NORM_MOMENTUM)
        print("[I] BATCH_NORM_EPSILON:", self.BATCH_NORM_EPSILON)
        print("[I] DENSE_UNITS:", self.DENSE_UNITS)
        print("[I] KERNEL_REGULARIZER_L2:", self.KERNEL_REGULARIZER_L2)
        print("[I] ACTIVITY_REGULARIZER_L1:", self.ACTIVITY_REGULARIZER_L1)
        print("[I] BIAS_REGULARIZER_L1:", self.BIAS_REGULARIZER_L1)
        print("[I] DROPOUT_RATE:", self.DROPOUT_RATE)
        print("[I] DROPOUT_SEED:", self.DROPOUT_SEED)
        print("[I] LEARNING_RATE:", self.LEARNING_RATE)
        print("[I] FUNC_ACTIVATION_RELU:", self.FUNC_ACTIVATION_RELU)
        print("[I] FUNC_ACTIVATION_SOFTMAX:", self.FUNC_ACTIVATION_SOFTMAX)
        print("[I] FUNC_LOSS:", self.FUNC_LOSS)
        print("[I] LOSS_METRIC:", self.LOSS_METRIC)