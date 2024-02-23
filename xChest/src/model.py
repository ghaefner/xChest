from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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