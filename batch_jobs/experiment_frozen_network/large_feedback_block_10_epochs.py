import sys
sys.path.append('../../')
import tensorflow as tf
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen4BlockTo1Block
from training_procedure import run_training

path_to_datasets = '../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
models = [VGG16FeedbackFrozen4BlockTo1Block, ]
model_names = ["Feedback_4block_to_1block_10_epochs", ]

if __name__ == "__main__":
    
    for model, model_name in zip(models, model_names):
        run_training(path_to_datasets=path_to_datasets, model_class=model, model_name=model_name, epochs=10)