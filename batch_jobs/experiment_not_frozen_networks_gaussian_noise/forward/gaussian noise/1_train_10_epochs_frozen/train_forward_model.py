import tensorflow_datasets as tfds
import numpy as np
import os
import sys
import json
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import json
from collections import defaultdict
from skimage import io
from skimage.color import rgb2gray
#from skimage.color import rgb2gray, rgb2grey
from scipy.ndimage.filters import gaussian_filter
from tensorflow.nn import local_response_normalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization, Lambda, GlobalAveragePooling2D, Dense, Dropout, Flatten, Add, UpSampling3D, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import Input, Model
from tensorflow import keras 
from skimage.util import random_noise
import imageio
import math
import random
import time
#from numba import cuda 
import os

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/opt/bwhpc/common/devel/cuda/11.8"

def append_to_file(file_name, val):
  f = open(file_name, "a")
  f.write("{0}\n".format(str(val)))
  f.close()

def gaussian_noise(image):
    row,col,ch= image.shape
    mean = 0
    var = 0.01
    sigma = var**0.4
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return noisy


def tf_gaussian_noise(x, y):
    x = tf.py_function(gaussian_noise, [x], np.float32)
    return x, y


def salt_pepper_noise(img):
    pad = 150
    show = 1
    noise = np.random.randint(pad, size = (img.shape[0], img.shape[1], 3))
    img = np.where(noise == 0, 0, img)
    img = np.where(noise == (pad-1), 1, img)
    noise = noise / 255
    return noise + img


def tf_salt_pepper_noise(x, y):
    x = tf.py_function(salt_pepper_noise, [x], np.float32)
    return x, y

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

    path_to_datasets = '../../../../create_datasets/normalize_dataset/saved_datasets/'
    train_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "train_ds_without_batching"))
    test_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "test_ds_without_batching"))
    valid_ds_prepared_without_batch = tf.data.Dataset.load(os.path.join(path_to_datasets, "valid_ds_without_batching"))

    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)

    batch_size = 64
    num_classes = 16
    img_size = 224
    input_shape = (None, img_size, img_size, 3) 
    epochs= 10


    with strategy.scope():
        train_ds = train_ds_prepared_without_batch.map(tf_gaussian_noise).shuffle(10000).batch(batch_size).prefetch(4)
        valid_ds = valid_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        test_ds = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        #test_ds_gaussian_noise = test_ds_prepared_without_batch.map(tf_gaussian_noise).batch(batch_size).prefetch(4)
        #test_ds_salt_pepper_noise = test_ds_prepared_without_batch.map(tf_salt_pepper_noise).batch(batch_size).prefetch(4)


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
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        start_time = time.time()
        history = model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[model_checkpoint_callback], verbose=1) 
        #model.load_weights(checkpoint_filepath)
        execution_time = time.time() - start_time
        results = model.evaluate(test_ds, batch_size=batch_size)

    file_res_forward = os.path.join("./results_forward_model.txt")
    if os.path.exists(file_res_forward):
      os.remove(file_res_forward)
    f = open(file_res_forward, "x")
    for v in ["test loss", results[0], "test acc", results[1]]:
        append_to_file(file_res_forward, v)    

    history_dict = history.history
    # Save it under the form of a json file
    history_path = "./history_forward.json"
    if os.path.exists(history_path):
      os.remove(history_path)
    f = open(history_path, "x")
    json.dump(history_dict, open(history_path, 'w'))

