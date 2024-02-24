from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from config import BATCH_SIZE, IMG_SIZE, IMG_SHAPE, PATH_MODEL_FOLDER, HyperPars
import time
from pickle import dump as pkl_dump
from pickle import load as pkl_load

def split_train_data(dict_folder):
    print("[I] Splitting train data.")

    train_df, dummy_df = train_test_split(dict_folder['train'], train_size=0.8, shuffle= True, random_state= 42)
    valid_df, test_df= train_test_split(dummy_df, train_size= 0.6, shuffle= True, random_state= 42)

    print("[I] Done.")
    return train_df, test_df, valid_df

def generate_images(train_df, test_df, valid_df):
    print("[I] Generating images fom file list.")

    def scalar(img):
        return img
    
    # Define ImageDataGenerator for data augmentation
    train_datagen = ImageDataGenerator(preprocessing_function=scalar)
    test_datagen = ImageDataGenerator(preprocessing_function=scalar)
    valid_datagen = ImageDataGenerator( preprocessing_function=scalar)
    
    # Create data generators
    train_gen = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepaths',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepaths',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)
    
    valid_gen = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filepaths',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)

    print("[I] Done.")
   
    return train_gen, test_gen, valid_gen


def initialize_model(train_gen, hyper_params=HyperPars()):
    print("[I] Initializing Model.")
    hyper_params.print_info()

    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    num_class = len(classes)

    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_shape=IMG_SHAPE, pooling='max')
    print("[I] Base Model Output Shape:", base_model.output_shape)

    batch_norm = BatchNormalization(axis=hyper_params.BATCH_NORM_AXIS, momentum=hyper_params.BATCH_NORM_MOMENTUM, epsilon=hyper_params.BATCH_NORM_EPSILON)
    dense_layer = Dense(hyper_params.DENSE_UNITS, 
                        kernel_regularizer=regularizers.l2(hyper_params.KERNEL_REGULARIZER_L2), 
                        activity_regularizer=regularizers.l1(hyper_params.ACTIVITY_REGULARIZER_L1),
                        bias_regularizer=regularizers.l1(hyper_params.BIAS_REGULARIZER_L1), 
                        activation=hyper_params.FUNC_ACTIVATION_RELU)
    print("[I] Shape after Dense Layer: (None, " + str(hyper_params.DENSE_UNITS) + ")")

    dropout_layer = Dropout(rate=hyper_params.DROPOUT_RATE, seed=hyper_params.DROPOUT_SEED)
    output_layer = Dense(num_class, activation=hyper_params.FUNC_ACTIVATION_SOFTMAX)


    model = Sequential([
        base_model,
        batch_norm,
        dense_layer,
        dropout_layer,
        output_layer
    ])

    print("[I] Compiling Model.")
    model.compile(Adamax(learning_rate=hyper_params.LEARNING_RATE), loss=hyper_params.FUNC_LOSS, metrics=[hyper_params.LOSS_METRIC])
    print("[I] Done.")

    return model


def fit_model(model, train_gen, valid_gen, epochs = 10):
    print("[I] Starting Model Fitting.")
    start_time = time.time()
    history = model.fit(
        x= train_gen, 
        epochs = epochs, 
        verbose = 1, 
        validation_data = valid_gen,
        validation_steps = None, 
        shuffle = False
    )
    stop_time = time.time()
    print(f'[I] Model Fitting finished in {stop_time-start_time: .2f} Seconds.')
    print("[I] Done.")
    return history

def save_history(history, filename="output_model.pkl"):
    """
    Save the training history to a file.
    
    Parameters:
        history (History object): The training history returned by the fit method.
        filename (str): The name of the file to save the history to.
    """
    filename = PATH_MODEL_FOLDER+filename
    with open(filename, 'wb') as file:
        pkl_dump(history.history, file)

def load_history(filename):
    """
    Load the training history from a file.
    
    Parameters:
        filename (str): The name of the file containing the saved history.
    
    Returns:
        dict: The loaded training history.
    """
    filename = PATH_MODEL_FOLDER+filename
    with open(filename, 'rb') as file:
        history = pkl_load(file)
    return history