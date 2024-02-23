from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from config import BATCH_SIZE, IMG_SHAPE

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
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepaths',
        y_col='label',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
    
    valid_gen = valid_datagen.flow_from_dataframe(
        dataframe=valid_df,
        x_col='filepaths',
        y_col='label',
        target_size=IMG_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    print("[I] Done.")
   
    return train_gen, test_gen, valid_gen


def initialize_model(train_gen):
    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    num_class = len(classes)

    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, weights='imagenet', input_shape=IMG_SHAPE, pooling='max')
    print("Shape of base model output:", base_model.output.shape)
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.4, seed=75),
        Dense(num_class, activation='softmax')
    ])
    model.compile(Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    '''
    # Run dummy model for model summary
    dummy_data = tf.zeros((1, *IMG_SHAPE))
    dummy_labels = tf.zeros((1, num_class))
    model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)

    model.summary()
    '''
    return model