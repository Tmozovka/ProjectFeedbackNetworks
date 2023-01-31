import numpy as np
import tensorflow as tf

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