import sys
import json
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import json
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization, Lambda, GlobalAveragePooling2D, Dense, Dropout, Flatten, Add, UpSampling3D, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow import keras
import time
import os
sys.path.append('../')
from src.DatasetNoise import tf_gaussian_noise, tf_salt_pepper_noise

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"

def append_to_file(file_name, val):
  f = open(file_name, "a")
  f.write("{0}\n".format(str(val)))
  f.close()

batch_size = 64
num_classes = 16
img_size = 224
input_shape = (None, img_size, img_size, 3)
epochs= 40

class FeedforwardModel(keras.Model):
  def __init__(self, **kwargs):
    super(FeedforwardModel, self).__init__(name="FFmodel", **kwargs)
    VGG = VGG16(weights='imagenet', include_top=False)
    self.VGG_without_maxpooling = keras.Model(VGG.input, VGG.layers[-2].output)
    self.VGG_without_maxpooling.trainable = False
    self.flatten = Flatten()
    self.dense = Dense(256, activation= 'relu')
    self.dropout = Dropout(0.5)
    self.maxpooling = MaxPooling2D(pool_size=(2,2), strides= (2,2))
    self.output_layer = Dense(num_classes, activation='softmax')

  def call(self, inputs):
    x = self.VGG_without_maxpooling(inputs)
    x = self.maxpooling(x)
    x = self.flatten(x)
    x = self.dense(x) 
    x = self.dropout(x)
    x = self.output_layer(x)
    return x

if __name__ == "__main__":

    path_to_datasets = '../../../create_datasets/normalize_dataset/saved_datasets/'
    train_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "train_ds_without_batching"))
    test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))
    valid_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "valid_ds_without_batching"))

    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)



    def resize(x, y):
        #x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.resize(x, (img_size, img_size))
        return x, y

    with strategy.scope():
        train_ds = train_ds_prepared_without_batch.shuffle(10000).batch(batch_size).prefetch(4)
        valid_ds = valid_ds_prepared_without_batch.batch(batch_size).prefetch(4)
        test_ds = test_ds_prepared_without_batch.batch(batch_size).prefetch(4)
        test_ds_gaussian_noise = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        test_ds_salt_pepper_noise = test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)


    checkpoint_filepath = './tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    with strategy.scope():
        sample = next(iter(test_ds))[0]
        model = FeedforwardModel()
        model.build(input_shape)
        model(sample)
        model.summary(show_trainable=True)

    with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        start_time = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[model_checkpoint_callback], verbose=1) 
        #model.load_weights(checkpoint_filepath)
        execution_time = time.time() - start_time
        results = model.evaluate(test_ds, batch_size=batch_size)
        results_gaussian_noise = model.evaluate(test_ds_gaussian_noise, batch_size=batch_size)
        results_salt_pepper_noise = model.evaluate(test_ds_salt_pepper_noise, batch_size=batch_size)

    file_res_forward = os.path.join("results_forward_model.txt")
    if os.path.exists(file_res_forward):
      os.remove(file_res_forward)
    f = open(file_res_forward, "x")
    for v in ["test loss", results[0], "test acc", results[1],\
             "test loss gaussian", results_gaussian_noise[0], "test acc gaussian", results_gaussian_noise[1],\
             "test loss salt_pepper", results_salt_pepper_noise[0], "test acc salt_pepper", results_salt_pepper_noise[1]]:
        append_to_file(file_res_forward, v)    

    history_dict = history.history
    history_path = "history_forward.json"
    if os.path.exists(history_path):
      os.remove(history_path)
    f = open(history_path, "x")
    json.dump(history_dict, open(history_path, 'w'))

