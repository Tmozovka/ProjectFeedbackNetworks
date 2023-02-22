import sys
import os
sys.path.append('../../')
import tensorflow as tf
import keras
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen4BlockTo1Block
from src.models.ForwardModelBuilder import VGG16Custom5BlockNotFrozen
from src.DatasetNoise import tf_gaussian_noise, tf_salt_pepper_noise
from src.Evaluation import get_confusion_matrix_for_model_and_data, visualize_confusion_matrix


batch_size = 8
num_classes = 16
img_size = 224
input_shape = (None, img_size, img_size, 3) 

path_to_datasets = '../../../../experiments/create_datasets/normalize_dataset/saved_datasets/'

test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))

models = {
"frozenVGG-Fb_4bl_to_1bl": {
    "model_class": VGG16FeedbackFrozen4BlockTo1Block,
    "checkpoint_filepath":"../../models/frozenVGG16/Feedback_4block_to_1block/checkpoint"
                                         },
"frozenVGGGaus-Fb_4bl_to_1bl": {
    "model_class": VGG16FeedbackFrozen4BlockTo1Block,
    "checkpoint_filepath":"../../models/frozenVGG16TrainGaussianNoise/Feedback_4block_to_1block/checkpoint"
                                         },
"frozenVGG-Fr_5_bl_not_froz": {
    "model_class": VGG16Custom5BlockNotFrozen,
    "checkpoint_filepath":"../../models/frozenVGG16/Forward_5block_not_frozen/checkpoint"
                                         },
"frozenVGGGaus-Fr_5_bl_not_froz": {
    "model_class": VGG16Custom5BlockNotFrozen,
    "checkpoint_filepath":"../../models/frozenVGG16TrainGaussianNoise/Forward_5block_not_frozen/checkpoint"
                                         },
}
datasets = {"without_noise": test_ds_prepared_without_batch.batch(batch_size).prefetch(4),
           "gaussian_noise": test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4),
           "salt_pepper_noise": test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)}


def build_model(model_class, model_checkpoint_filepath):
    sample = next(iter(test_ds))[0]
    model = model_class()
    model.build(input_shape)
    model(sample)
    model.summary(show_trainable=True)
    model.load_weights(model_checkpoint_filepath)
    return model


if __name__ == "__main__":
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    all_result = dict()
    with strategy.scope():
        for model_name, model_inf in models.items():
            for ds_name, test_ds in datasets.items():
                X_test = list(map(lambda x: x[0], test_ds))
                y_test = list(map(lambda x: x[1], test_ds))
                X_test = X_test[:len(X_test)-1]#[0]
                y_test = y_test[:len(y_test)-1]#[0]
                
                model_class = model_inf["model_class"]
                model_checkpoint_filepath = model_inf["checkpoint_filepath"]
                model = build_model(model_class, model_checkpoint_filepath)
                
                confusion_matrix = get_confusion_matrix_for_model_and_data(model, X_test, y_test)
                path_to_save = f"confusion_matr_{model_name}_on_{ds_name}.png"
                visualize_confusion_matrix(confusion_matrix, model_name = model_name, dataset_name = ds_name, path_to_save = path_to_save)