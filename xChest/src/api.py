import os
from pandas import DataFrame, Series, concat

from xChest.config import Path

def list_files(folder_path) -> DataFrame:
    """
    List files in the specified folder and create a DataFrame with file paths and corresponding labels.

    Args:
        folder_path (str): Path to the folder containing the files.

    Returns:
        DataFrame: DataFrame containing file paths and labels.
    """
    filepaths = []
    labels = []
    folds = os.listdir(folder_path)

    for fold in [fold for fold in folds if not fold.startswith(".")]:
        f_path = os.path.join(folder_path , fold)
        filelists = os.listdir(f_path)
        
        for file in filelists:
            filepaths.append(os.path.join(f_path , file))
            labels.append(fold)
            
    Fseries = Series(filepaths , name='filepaths')
    Lseries = Series(labels , name='label')
    df = concat([Fseries, Lseries], axis=1)

    return df

def list_subfolders(path_base=Path.BASE, path_sub=Path.SUBFOLDERS):
    """
    List files in the subfolders of the specified base folder and create a dictionary of DataFrames.

    Args:
        path_base (str, optional): Path to the base folder. Defaults to PATH_BASE.
        path_sub (list of str, optional): List of subfolder paths relative to the base folder. Defaults to PATH_SUB.

    Returns:
        dict: Dictionary containing subfolder names as keys and corresponding DataFrames as values.
    """
    dfs = [list_files(path_base+suffix) for suffix in path_sub]

    dict_folder = {subfolder.rstrip("/"): df for subfolder, df in zip(path_sub, dfs)}

    return dict_folder
