import sys
sys.path.append('../../../')
from src.models.FeedbackModelBuilder import VGG16FeedbackFrozen4BlockTo4Block,\
                                            VGG16FeedbackFrozen5BlockTo5Block,\
                                            VGG16FeedbackFrozen4BlockTo1Block,\
                                            VGG16FeedbackFrozen5BlockTo3Block,\
                                            VGG16FeedbackFrozen5BlockTo4Block

from src.models.ForwardModelBuilder import VGG16CustomFrozen
from src.Evaluation import run_mcnemar_test
from src.DatasetNoise import tf_gaussian_noise, tf_salt_pepper_noise
import tensorflow as tf
import itertools
import os

path_to_datasets = '../../../experiments/create_datasets/normalize_dataset/saved_datasets/'
test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))

batch_size = 256
num_classes = 16
img_size = 224
input_shape = (None, img_size, img_size, 3)
test_ds = test_ds_prepared_without_batch.batch(batch_size).prefetch(4)
test_ds_gaussian_noise = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
test_ds_salt_pepper_noise = test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)


name_model_path = {
"frozenVGG16-Feedback_5block_to_3block": {
    "model_class": VGG16FeedbackFrozen5BlockTo3Block,
    "checkpoint_filepath":"../models/frozenVGG16/Feedback_5block_to_3block/checkpoint"
                                         },
"frozenVGG16-Feedback_4block_to_4block": {
    "model_class": VGG16FeedbackFrozen4BlockTo4Block,
    "checkpoint_filepath":"../models/frozenVGG16/Feedback_4block_to_4block/checkpoint"
                                         },
"frozenVGG16-Feedback_4block_to_1block": {
    "model_class": VGG16FeedbackFrozen4BlockTo1Block,
    "checkpoint_filepath":"../models/frozenVGG16/Feedback_4block_to_1block/checkpoint"
                                         },
"frozenVGG16-Feedback_5block_to_4block": {
    "model_class": VGG16FeedbackFrozen5BlockTo4Block,
    "checkpoint_filepath":"../models/frozenVGG16/Feedback_5block_to_4block/checkpoint"
                                         },
"frozenVGG16-Feedback_5block_to_5block": {
    "model_class": VGG16FeedbackFrozen5BlockTo5Block,
    "checkpoint_filepath":"../models/frozenVGG16/Feedback_5block_to_5block/checkpoint"
                                         },
"frozenVGG16-Forward": {
    "model_class": VGG16CustomFrozen,
    "checkpoint_filepath":"../models/frozenVGG16/Forward/checkpoint"
                                         }
}

def build_model(model_class, model_checkpoint_filepath):
    sample = next(iter(test_ds))[0]
    model = model_class()
    model.build(input_shape)
    model(sample)
    model.summary(show_trainable=True)
    model.load_weights(model_checkpoint_filepath)
    return model

if __name__ == "__main__":
    for model_1, model_2 in itertools.combinations([(k,v) for k,v in name_model_path.items()], 2):
        print(model_1)
        print(model_2)
        model_1_class = model_1[1]["model_class"]
        model_1_checkpoint_filepath = model_1[1]["checkpoint_filepath"]
        model_1_name = model_1[0]

        model_2_class = model_2[1]["model_class"]
        model_2_checkpoint_filepath = model_2[1]["checkpoint_filepath"]
        model_2_name = model_2[0]

        model_1 = build_model(model_1_class, model_1_checkpoint_filepath)
        model_2 = build_model(model_2_class, model_2_checkpoint_filepath)

        print(run_mcnemar_test(model_1, model_2, test_ds))
        break