from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization, Lambda, GlobalAveragePooling2D, Dense, Dropout, Flatten, Add, UpSampling3D, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import Input, Model
from tensorflow import keras 
import time


num_classes = 16

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
    
class VGG16Custom5BlockNotFrozen(keras.Model):
    def __init__(self, **kwargs):
        super(VGG16Custom5BlockNotFrozen, self).__init__(name="FFmodel", **kwargs)
        VGG = VGG16(weights='imagenet', include_top=False)
        self.VGG_before_feedback = keras.Model(VGG.input, VGG.layers[-5].output)
        self.VGG_before_feedback.trainable = False
        
        self.maxpooling1 = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        self.conv1 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv2 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv3 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        
        self.flatten = Flatten()
        self.dense = Dense(256, activation= 'relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        x = inputs
        x = self.VGG_before_feedback(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpooling1(x)
        x = self.flatten(x)
        x = self.dense(x) 
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
class VGG16Custom5and4BlocksNotFrozen(keras.Model):
    def __init__(self, **kwargs):
        super(VGG16Custom5and4BlocksNotFrozen, self).__init__(name="FFmodel", **kwargs)
        VGG = VGG16(weights='imagenet', include_top=False)
        self.VGG_before_feedback = keras.Model(VGG.input, VGG.layers[-9].output)
        self.VGG_before_feedback.trainable = False
        
        self.maxpooling1 = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        self.conv1 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv2 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv3 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv4 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv5 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.conv6 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        
        self.upsampling = UpSampling3D(size=2)
        
        self.flatten = Flatten()
        self.dense = Dense(256, activation= 'relu')
        self.dropout = Dropout(0.5)
        self.maxpooling2 = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        self.output_layer = Dense(num_classes, activation='softmax')

        self.project_conv1 = Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu')

    def call(self, inputs):
        x = inputs
        x = self.VGG_before_feedback(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpooling1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.maxpooling2(x)
        x = self.flatten(x)
        x = self.dense(x) 
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
class VGG16Custom5and4and3BlocksNotFrozen(keras.Model):
    def __init__(self, **kwargs):
        super(VGG16Custom5and4and3BlocksNotFrozen, self).__init__(name="FFmodel", **kwargs)
        VGG = VGG16(weights='imagenet', include_top=False)
        self.VGG_before_feedback = keras.Model(VGG.input, VGG.layers[-13].output)
        self.VGG_before_feedback.trainable = False
        
        self.b3_conv1 = Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b3_conv2 = Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b3_conv3 = Conv2D(256, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b3_maxpooling = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        self.b4_conv1 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b4_conv2 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b4_conv3 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b4_maxpooling = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        self.b5_conv1 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b5_conv2 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b5_conv3 = Conv2D(512, kernel_size=(3,3), padding= 'same', activation= 'relu')
        self.b5_maxpooling = MaxPooling2D(pool_size=(2,2), strides= (2,2))
        
        self.upsampling = UpSampling3D(size=4)
        
        self.flatten = Flatten()
        self.dense = Dense(256, activation= 'relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(num_classes, activation='softmax')

        self.project_conv1 = Conv2D(128, kernel_size=(3,3), padding= 'same', activation= 'relu')

    def call(self, inputs):
        x = inputs
        x = self.VGG_before_feedback(x)
        x = self.b3_conv1(x)
        x = self.b3_conv2(x)
        x = self.b3_conv2(x)
        x = self.b3_maxpooling(x)
        x = self.b4_conv1(x)
        x = self.b4_conv2(x)
        x = self.b4_conv2(x)
        x = self.b4_maxpooling(x)
        x = self.b5_conv1(x)
        x = self.b5_conv2(x)
        x = self.b5_conv2(x)

        x = self.b5_maxpooling(x)
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