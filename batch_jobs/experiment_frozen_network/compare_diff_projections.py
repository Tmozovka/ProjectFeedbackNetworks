import sys
sys.path.append('../../')
import tensorflow as tf
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen5BlockTo3Block3ProjLayers, VGG16FeedbackFrozen5BlockTo3Block2ProjLayers
from training_procedure import run_training

path_to_datasets = '../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
#models = [VGG16FeedbackFrozen4BlockTo4Block, VGG16FeedbackFrozen5BlockTo5Block, VGG16FeedbackFrozen5BlockTo4Block]
#model_names = ["Feedback_4block_to_4block", "Feedback_5block_to_5block", "Feedback_5block_to_4block"]
models = [VGG16FeedbackFrozen5BlockTo3Block3ProjLayers, VGG16FeedbackFrozen5BlockTo3Block2ProjLayers]
model_names = ["Feedback_5block_to_3block_3_projection_layers", "Feedback_5block_to_3block_2_projection_layers"]

if __name__ == "__main__":
    
    for model, model_name in zip(models, model_names):
        run_training(path_to_datasets=path_to_datasets, model_class=model, model_name=model_name)