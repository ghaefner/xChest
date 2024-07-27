from datetime import datetime
import logging

class Path:
    BASE = "D:/Projekte/2024/Chest_Xray_Images/"
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
        logging.info("Hyperparameters used:")
        logging.info("BATCH_NORM_AXIS: %s", self._BATCH_NORM_AXIS)
        logging.info("BATCH_NORM_MOMENTUM: %s", self._BATCH_NORM_MOMENTUM)
        logging.info("BATCH_NORM_EPSILON: %s", self._BATCH_NORM_EPSILON)
        logging.info("DENSE_UNITS: %s", self._DENSE_UNITS)
        logging.info("KERNEL_REGULARIZER_L2: %s", self._KERNEL_REGULARIZER_L2)
        logging.info("ACTIVITY_REGULARIZER_L1: %s", self._ACTIVITY_REGULARIZER_L1)
        logging.info("BIAS_REGULARIZER_L1: %s", self._BIAS_REGULARIZER_L1)
        logging.info("DROPOUT_RATE: %s", self._DROPOUT_RATE)
        logging.info("DROPOUT_SEED: %s", self._DROPOUT_SEED)
        logging.info("LEARNING_RATE: %s", self._LEARNING_RATE)
        logging.info("FUNC_ACTIVATION_RELU: %s", self._FUNC_ACTIVATION_RELU)
        logging.info("FUNC_ACTIVATION_SOFTMAX: %s", self._FUNC_ACTIVATION_SOFTMAX)
        logging.info("FUNC_LOSS: %s", self._FUNC_LOSS)
        logging.info("LOSS_METRIC: %s", self._LOSS_METRIC)

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