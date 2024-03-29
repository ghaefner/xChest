import os
import time

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from pickle import dump as pkl_dump, load as pkl_load
from numpy import argmax

from xChest.config import BATCH_SIZE, IMG_SIZE, IMG_SHAPE, Path, HyperPars, CURRENT_DATE
from xChest.src.plot import plot_model_accuracy, save_plot
from xChest.src.api import list_subfolders


def split_train_data(dict_folder):
    """
    Splits the training data into train, test, and validation sets.

    Parameters:
        dict_folder (dict): A dictionary containing the data to be split, with keys 'train'.

    Returns:
        tuple: A tuple containing three DataFrames representing train, test, and validation sets, respectively.
    """
    print("[I] Splitting train data.")

    train_df, dummy_df = train_test_split(dict_folder['train'], train_size=0.8, shuffle= True, random_state= 42)
    valid_df, test_df= train_test_split(dummy_df, train_size= 0.6, shuffle= True, random_state= 42)

    print("[I] Done.")
    return train_df, test_df, valid_df

def generate_images(train_df, test_df, valid_df):
    """
    Generates image data generators for training, testing, and validation from the provided dataframes.

    Parameters:
        train_df (DataFrame): DataFrame containing training data.
        test_df (DataFrame): DataFrame containing testing data.
        valid_df (DataFrame): DataFrame containing validation data.

    Returns:
        tuple: A tuple containing three ImageDataGenerator objects for training, testing, and validation, respectively.
    """
    print("[I] Generating images from file list.")

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
    """
    Initializes and compiles a Keras model for image classification.

    Parameters:
        train_gen (DirectoryIterator): Generator for training data.
        hyper_params (HyperPars, optional): Object containing hyperparameters (default is HyperPars()).

    Returns:
        tf.keras.Model: Compiled Keras model for image classification.
    """
    print("[I] Initializing Model.")
    hyper_params.print_info()

    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    num_class = len(classes)

    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_shape=IMG_SHAPE, pooling='max')
    print("[I] Base Model Output Shape:", base_model.output_shape)

    batch_norm = BatchNormalization(axis=hyper_params._BATCH_NORM_AXIS, momentum=hyper_params._BATCH_NORM_MOMENTUM, epsilon=hyper_params.BATCH_NORM_EPSILON)
    dense_layer = Dense(hyper_params._DENSE_UNITS, 
                        kernel_regularizer=regularizers.l2(hyper_params._KERNEL_REGULARIZER_L2), 
                        activity_regularizer=regularizers.l1(hyper_params._ACTIVITY_REGULARIZER_L1),
                        bias_regularizer=regularizers.l1(hyper_params._BIAS_REGULARIZER_L1), 
                        activation=hyper_params._FUNC_ACTIVATION_RELU)
    print("[I] Shape after Dense Layer: (None, " + str(hyper_params._DENSE_UNITS) + ")")

    dropout_layer = Dropout(rate=hyper_params._DROPOUT_RATE, seed=hyper_params._DROPOUT_SEED)
    output_layer = Dense(num_class, activation=hyper_params._FUNC_ACTIVATION_SOFTMAX)


    model = Sequential([
        base_model,
        batch_norm,
        dense_layer,
        dropout_layer,
        output_layer
    ])

    print("[I] Compiling Model.")
    model.compile(Adamax(learning_rate=hyper_params._LEARNING_RATE), loss=hyper_params._FUNC_LOSS, metrics=[hyper_params._LOSS_METRIC])
    print("[I] Done.")

    return model


def fit_model(model, train_gen, valid_gen, epochs = 10):
    """
    Fits the specified model using the provided generators for training and validation.

    Parameters:
        model (tf.keras.Model): The Keras model to fit.
        train_gen (DirectoryIterator): The generator for training data.
        valid_gen (DirectoryIterator): The generator for validation data.
        epochs (int): The number of epochs to train the model (default is 10).

    Returns:
        History object: The training history.
    """
    print("[I] Starting Model Fitting.")
    history = model.fit(
        x= train_gen, 
        epochs = epochs, 
        verbose = 1, 
        validation_data = valid_gen,
        validation_steps = None, 
        shuffle = False
    )
    print("[I] Done.")
    return history

def save_history(history, filename="output_model.pkl"):
    """
    Save the training history to a file.
    
    Parameters:
        history (History object): The training history returned by the fit method.
        filename (str): The name of the file to save the history to.
    """
    filename = Path.MODELS+filename
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
    filename = Path.MODELS+filename
    with open(filename, 'rb') as file:
        history = pkl_load(file)
    return history

def extract_model_accuracy(history):
    """
    Extract model accuracy from saved model history.
    
    Parameters:
        history (model.history): The name of the file containing the saved history.
    
    Returns:
        tuple: Training accuracy, loss and validation accuracy, loss.
    """
    
    train_acc = history['accuracy']
    train_loss = history['loss']

    val_acc = history['val_accuracy']
    val_loss = history['val_loss']

    return train_acc, train_loss, val_acc, val_loss

def print_training_accuracy(history):
    tr_acc, tr_loss, val_acc, val_loss = extract_model_accuracy(history)

    print("-"*40)
    print(f'[I] Training accuracy: {tr_acc}.')
    print(f'[I] Training loss: {tr_loss}.')
    print(f'[I] Validation accuracy: {val_acc}.')
    print(f'[I] Training accuracy: {val_loss}.')
    print("-"*40)

def evaluate_model(model,train_gen, test_gen, valid_gen):
    """
    Evaluate model on generated image sets.
    
    Parameters:
        model: The name of the  model.
        train_gen: Generated training images.
        test_gen: Generated test images.
        valid_gen: Generated validation images.
    
    Returns:
        None: Only prints model scores.
    """

    train_score = model.evaluate(train_gen, steps=16, verbose=1)
    test_score = model.evaluate(test_gen, steps=16, verbose=1)
    valid_score = model.evaluate(valid_gen, steps=16, verbose=1)

    print("[I] Model Evaluation.")
    print('-' * 20)
    print("[I] Train Loss: ", train_score[0])
    print("[I] Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("[I] Validation Loss: ", valid_score[0])
    print("[I] Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("[I] Test Loss: ", test_score[0])
    print("[I] Test Accuracy: ", test_score[1])
    print('-' * 20)


def create_confusion_matrix(model, test_gen):
    """
    Create a confusion matrix from test images and prints the classification report.
    
    Parameters:
        model: The name of the  model.
        test_gen: Generated test images.
    
    Returns:
        cm: Confusion matrix.
        classes: Different classes for the confusion matrix from the list of images.
    
    """
    preds = model.predict_generator(test_gen)
    y_pred = argmax(preds, axis=1)

    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())
    cm = confusion_matrix(y_true=test_gen.classes, y_pred=y_pred)

    print(classification_report(test_gen.classes, y_pred, target_names=classes))

    return cm, classes

def run_model(dict_folder, model_output_name, hyper_params=HyperPars()):
    """
    Run the model training process using the provided data, hyperparameters, and model name.

    Args:
        dict_folder (dict): Dictionary containing subfolder names as keys and corresponding DataFrames as values.
        model_output_name (str): Name of the model output.
        hyper_params (HyperPars, optional): Object containing hyperparameters for model initialization. 
            Defaults to HyperPars().

    Returns:
        dict: History object containing training metrics.
    """    
    df_train, df_test, df_valid = split_train_data(dict_folder)
    train_gen, _, valid_gen = generate_images(df_train, df_test, df_valid)

    model = initialize_model(train_gen=train_gen, hyper_params=hyper_params)

    history = fit_model(model, train_gen=train_gen, valid_gen=valid_gen, epochs=hyper_params._EPOCHS)
            
    print(f'[I] Saving Model History {model_output_name} to Model Folder.')
    save_history(history=history, filename=model_output_name)



class TaskModel:
    """A class representing a task for model training and evaluation."""

    def __init__(self):
        """
        Initialize the TaskModel instance.

        Args:
            dict_folder (str): Path to the folder containing data for model training.
        """
        self.dict_folder = list_subfolders()
        self.model_output_name = "V"+str(CURRENT_DATE)+"_Model.pkl"

    def run(self, hyper_params=HyperPars()):
        """
        Run the model training task.

        Args:
            hyper_params (HyperPars, optional): Hyperparameters for model training. 
                Defaults to HyperPars().
        """
        start_time = time.time()
        print("[I] Starting Model Training Task.")

        if os.path.exists(os.path.join(Path.MODELS, self.model_output_name)):
            print(f'[I] Model {self.model_output_name} already exists.')
            print("[I] Loading Model History.")
            load_history(self.model_output_name)
        
        else:
            print(f'[I] Model {self.model_output_name} does not exist. Running Model.')
            run_model(dict_folder=self.dict_folder, 
                      model_output_name=self.model_output_name, 
                      hyper_params=hyper_params)

        stop_time = time.time()
        print(f'[I] Task finished in {stop_time-start_time: .3f} Seconds.')

    def get_accuracy(self):
        """
        Evaluate the model accuracy and plot the results.

        Raises:
            ValueError: If the model file does not exist.
        """
        if os.path.exists(os.path.join(Path.MODELS, self.model_output_name)):
            print(f'[I] Evaluating model {self.model_output_name}.')
            history = load_history(self.model_output_name)
            print_training_accuracy(history)

            print("[I] Plotting model accuracy.")
            fig = plot_model_accuracy(*extract_model_accuracy(history=history))
            save_plot(obj=fig, output_path="Model_Accuracy.png")

        else:
            raise ValueError("[E] Model does not exist. Run model first.")