#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:19:50 2019

@author: RyanBansal
"""

# importing libraries
from keras.applications import ResNet50
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Flatten
from keras import backend as K 
from scipy import ndimage


img_width, img_height = 224, 224

train_data_dir = '/Users/byanbansal/Desktop/HotorBot/Train/'
validation_data_dir = '/Users/byanbansal/Desktop/HotorBot/Validate/'
nb_train_samples = 122
nb_validation_samples = 22
epochs = 10
my_input_shape = (img_width, img_height, 3)
batch_size = 1

### Sequential Architecture ###
model = Sequential()

model.add(ResNet50(include_top = False,
                   weights = 'imagenet',
                   input_shape = my_input_shape))

model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False

### Functional API Architecture ###
input_layer = Input(shape = my_input_shape)

base_renet50_model = ResNet50(include_top = False, weights = 'imagenet')

# Freezing the ResNet50 layers
for layer in base_renet50_model.layers:
    layer.trainable = False

X = (base_renet50_model) (input_layer)
X = GlobalAveragePooling2D() (X)

predicting_layer = Dense(1, activation = 'sigmoid') (X)

model = Model(inputs = input_layer, outputs = predicting_layer)

# Compilation
model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

# Summary
model.summary()

# Train data generation object with augmentation.
train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                   rotation_range = 30,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   horizontal_flip = True)

# Validation data generator object without augmentation.
validation_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_directory(train_data_dir, 
                                                    target_size = (img_width, img_height),
                                                    batch_size = batch_size,
                                                    class_mode = 'binary')

validation_generator = validation_datagen.flow_from_directory(validation_data_dir,
                                                              target_size = (img_width, img_height),
                                                              batch_size = batch_size,
                                                              class_mode = 'binary')

# Fitting the model.
model.fit_generator(train_generator, 
                    steps_per_epoch = nb_train_samples // batch_size,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = nb_validation_samples // batch_size)

print('Model has been trained and is ready to use.')

model.save('model_saved2.h5')
print('Model saved to disk.') 



