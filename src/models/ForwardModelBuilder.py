from tensorflow import keras 
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization, Lambda, GlobalAveragePooling2D, Dense, Dropout, Flatten, Add, UpSampling3D, GlobalAveragePooling2D


class VGG16CustomFrozen(keras.Model):
    def __init__(self, **kwargs):
        super(VGG16CustomFrozen, self).__init__(name="FFmodel", **kwargs)
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

class VGG16Custom(keras.Model):
    def __init__(self, **kwargs):
        super(VGG16Custom, self).__init__(name="FFmodel", **kwargs)
        VGG = VGG16(weights='imagenet', include_top=False)
        self.VGG_without_maxpooling = keras.Model(VGG.input, VGG.layers[-2].output)
        self.VGG_without_maxpooling._name = "not_frozen_vgg16"
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