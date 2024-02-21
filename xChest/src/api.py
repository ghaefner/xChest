import os
from config import PATH_TO_TRAIN_FOLDER
from pandas import DataFrame, Series, concat

def list_files(folder_path) -> DataFrame:
    filepaths =[]
    labels = []
    folds = os.listdir(folder_path)

    for fold in folds:
        f_path = os.path.join(folder_path , fold)
        filelists = os.listdir(f_path)
        
        for file in filelists:
            filepaths.append(os.path.join(f_path , file))
            labels.append(fold)
            
    Fseries = Series(filepaths , name = 'filepaths')
    Lseries = Series(labels , name = 'label')
    df = concat([Fseries , Lseries] , axis = 1)

    return df

