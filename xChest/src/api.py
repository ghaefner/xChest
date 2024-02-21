import os
from config import PATH_BASE, PATH_SUB
from pandas import DataFrame, Series, concat

def list_files(folder_path) -> DataFrame:
    filepaths =[]
    labels = []
    folds = os.listdir(folder_path)

    for fold in [fold for fold in folds if not fold.startswith(".")]:
        f_path = os.path.join(folder_path , fold)
        filelists = os.listdir(f_path)
        
        for file in filelists:
            filepaths.append(os.path.join(f_path , file))
            labels.append(fold)
            
    Fseries = Series(filepaths , name = 'filepaths')
    Lseries = Series(labels , name = 'label')
    df = concat([Fseries , Lseries] , axis = 1)

    return df

def list_subfolders(path_base=PATH_BASE, path_sub=PATH_SUB):

    dfs = [list_files(path_base+suffix) for suffix in path_sub]
    return dfs