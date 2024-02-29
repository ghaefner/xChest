class Path:
    BASE = "D:/Download/chest_xray/"
    PLOTS = "xChest/plots/"
    MODELS = "xChest/models/"
    SUBFOLDERS = ["train/", "test/", "val/"]


IMG_SIZE = (244,244)
IMG_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)
BATCH_SIZE = 16
CURRENT_DATE = datetime.now().strftime('%Y%m%d')


class HyperPars:
    def __init__(self):
        self._BATCH_NORM_AXIS = -1
        self._BATCH_NORM_MOMENTUM = 0.99
        self._BATCH_NORM_EPSILON = 0.001
        self._DENSE_UNITS = 256
        self._KERNEL_REGULARIZER_L2 = 0.016
        self._ACTIVITY_REGULARIZER_L1 = 0.06
        self._BIAS_REGULARIZER_L1 = 0.006
        self._DROPOUT_RATE = 0.04
        self._DROPOUT_SEED = 75
        self._LEARNING_RATE = 0.0001
        self._FUNC_ACTIVATION_RELU = "relu"
        self._FUNC_ACTIVATION_SOFTMAX = "softmax"
        self._FUNC_LOSS = "categorical_crossentropy"
        self._LOSS_METRIC = "accuracy"
        self._EPOCHS = 10

    def print_info(self):
        print("[I] Hyperparameters used:")
        print("[I] BATCH_NORM_AXIS:", self._BATCH_NORM_AXIS)
        print("[I] BATCH_NORM_MOMENTUM:", self._BATCH_NORM_MOMENTUM)
        print("[I] BATCH_NORM_EPSILON:", self._BATCH_NORM_EPSILON)
        print("[I] DENSE_UNITS:", self._DENSE_UNITS)
        print("[I] KERNEL_REGULARIZER_L2:", self._KERNEL_REGULARIZER_L2)
        print("[I] ACTIVITY_REGULARIZER_L1:", self._ACTIVITY_REGULARIZER_L1)
        print("[I] BIAS_REGULARIZER_L1:", self._BIAS_REGULARIZER_L1)
        print("[I] DROPOUT_RATE:", self._DROPOUT_RATE)
        print("[I] DROPOUT_SEED:", self._DROPOUT_SEED)
        print("[I] LEARNING_RATE:", self._LEARNING_RATE)
        print("[I] FUNC_ACTIVATION_RELU:", self._FUNC_ACTIVATION_RELU)
        print("[I] FUNC_ACTIVATION_SOFTMAX:", self._FUNC_ACTIVATION_SOFTMAX)
        print("[I] FUNC_LOSS:", self._FUNC_LOSS)
        print("[I] LOSS_METRIC:", self._LOSS_METRIC)

    @property
    def BATCH_NORM_AXIS(self):
        return self._BATCH_NORM_AXIS

    @BATCH_NORM_AXIS.setter
    def BATCH_NORM_AXIS(self, value):
        self._BATCH_NORM_AXIS = value

    @property
    def BATCH_NORM_MOMENTUM(self):
        return self._BATCH_NORM_MOMENTUM

    @BATCH_NORM_MOMENTUM.setter
    def BATCH_NORM_MOMENTUM(self, value):
        self._BATCH_NORM_MOMENTUM = value

    @property
    def BATCH_NORM_EPSILON(self):
        return self._BATCH_NORM_EPSILON

    @BATCH_NORM_EPSILON.setter
    def BATCH_NORM_EPSILON(self, value):
        self._BATCH_NORM_EPSILON = value

    @property
    def DENSE_UNITS(self):
        return self._DENSE_UNITS

    @DENSE_UNITS.setter
    def DENSE_UNITS(self, value):
        self._DENSE_UNITS = value

    @property
    def KERNEL_REGULARIZER_L2(self):
        return self._KERNEL_REGULARIZER_L2

    @KERNEL_REGULARIZER_L2.setter
    def KERNEL_REGULARIZER_L2(self, value):
        self._KERNEL_REGULARIZER_L2 = value

    @property
    def ACTIVITY_REGULARIZER_L1(self):
        return self._ACTIVITY_REGULARIZER_L1

    @ACTIVITY_REGULARIZER_L1.setter
    def ACTIVITY_REGULARIZER_L1(self, value):
        self._ACTIVITY_REGULARIZER_L1 = value

    @property
    def BIAS_REGULARIZER_L1(self):
        return self._BIAS_REGULARIZER_L1

    @BIAS_REGULARIZER_L1.setter
    def BIAS_REGULARIZER_L1(self, value):
        self._BIAS_REGULARIZER_L1 = value

    @property
    def DROPOUT_RATE(self):
        return self._DROPOUT_RATE

    @DROPOUT_RATE.setter
    def DROPOUT_RATE(self, value):
        self._DROPOUT_RATE = value

    @property
    def DROPOUT_SEED(self):
        return self._DROPOUT_SEED

    @DROPOUT_SEED.setter
    def DROPOUT_SEED(self, value):
        self._DROPOUT_SEED = value

    @property
    def LEARNING_RATE(self):
        return self._LEARNING_RATE

    @LEARNING_RATE.setter
    def LEARNING_RATE(self, value):
        self._LEARNING_RATE = value

    @property
    def FUNC_ACTIVATION_RELU(self):
        return self._FUNC_ACTIVATION_RELU

    @FUNC_ACTIVATION_RELU.setter
    def FUNC_ACTIVATION_RELU(self, value):
        self._FUNC_ACTIVATION_RELU = value

    @property
    def FUNC_ACTIVATION_SOFTMAX(self):
        return self._FUNC_ACTIVATION_SOFTMAX

    @FUNC_ACTIVATION_SOFTMAX.setter
    def FUNC_ACTIVATION_SOFTMAX(self, value):
        self._FUNC_ACTIVATION_SOFTMAX = value

    @property
    def FUNC_LOSS(self):
        return self._FUNC_LOSS

    @FUNC_LOSS.setter
    def FUNC_LOSS(self, value):
       self._FUNC_LOSS = value

    @property
    def EPOCHS(self):
        return self._FUNC_LOSS

    @EPOCHS.setter
    def EPOCHS(self, value):
       self._EPOCHS = value