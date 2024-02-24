from src.api import list_subfolders
from src.model import TaskModel
from config import HyperPars

# Initialize Paramters
dict_folder = list_subfolders()
task_model = TaskModel(dict_folder=dict_folder)


# Run Model
task_model.run(hyper_params=HyperPars())

