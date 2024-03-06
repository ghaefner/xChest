from xChest.src.api import list_subfolders
from xChest.src.model import TaskModel
from xChest.config import HyperPars

# Initialize Paramters
task_model = TaskModel()


# Run Model
task_model.run(hyper_params=HyperPars())
task_model.get_accuracy()
