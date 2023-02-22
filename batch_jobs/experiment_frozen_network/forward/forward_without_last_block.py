import sys
sys.path.append('../../../')
import tensorflow as tf
from src.models.ForwardModelBuilder import VGG16Custom5BlockNotFrozen
from training_procedure import run_training

path_to_datasets = '../../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
models = [VGG16Custom5BlockNotFrozen, ]
model_names = ["Forward_5block_not_frozen", ]

if __name__ == "__main__":
    
    for model, model_name in zip(models, model_names):
        run_training(path_to_datasets=path_to_datasets, model_class=model, model_name=model_name)