import sys
sys.path.append('../../../')
import tensorflow as tf
from src.models.ForwardModelBuilder import VGG16Custom5and4and3BlocksNotFrozen, VGG16Custom5and4BlocksNotFrozen
from training_procedure import run_training

path_to_datasets = '../../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
#models = [VGG16Custom5and4and3BlocksNotFrozen, VGG16Custom5and4BlocksNotFrozen]
#model_names = ["Forward_5_4_3_blocks_not_frozen", "Forward_5_4_blocks_not_frozen"]
models = [VGG16Custom5and4BlocksNotFrozen, ]
model_names = ["Forward_5_4_blocks_not_frozen", ]

if __name__ == "__main__":
    
    for model, model_name in zip(models, model_names):
        run_training(path_to_datasets=path_to_datasets, model_class=model, model_name=model_name)