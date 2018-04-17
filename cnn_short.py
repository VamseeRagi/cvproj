from __future__ import print_function

import numpy as np
from scipy.io import loadmat
from IPython import embed
from keras import backend as K
K.clear_session()

import matplotlib.pyplot as plt
import keras


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau


import os

from cnn_models import *


# train_mat = loadmat("data.mat")
# test_mat = loadmat("temp.mat")

all_data = loadmat("data.mat")['X']


def get_train(all_data):
    training_subset_index = np.random.randint(1, 13233, 10)
    suspect_index = np.random.randint(0, 9, 1)
    suspect_index = training_subset_index[suspect_index[0]]
    training_subset = all_data[training_subset_index - 1, :, :, :]
    suspect = training_subset[suspect_index, :, :, :]

    y_array = np.zeros(shape=(11, 1))
    y_array[suspect_index + 1] = np.array([1])

    x_matrix = np.zeros(shape=(11, 100, 100, 3))

    x_matrix[0, :, :, :] = suspect
    x_matrix[1:, :, :, :] = training_subset

    return x_matrix, y_array


x_train, y_train = get_train(all_data)


batch_size = 100
num_classes = 10
epochs = 100
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'data2.h5'

y_train = keras.utils.to_categorical(y_train, num_classes)

w_reg = 1e-4


model = build_model([11,100,100,3], 1e-4, 10, opt={
  "model": "vgg", 
  "fc_after_cnn": True,
  "fc_after_cnn_dim": 200,
  "comparing_method": "mmul",
  "comparing_method_mmul_w_dim": 20,
  "comparing_method_mmul_use_original_target": True,
  "": ""
})
  
opt = keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.90, nesterov=False)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_train[0, :, :, :].reshape(1, 100, 100, 3)
x_train /= 255
x_test /= 255


def lrf(e):
  if e > 40: 
    return 1e-4
  if e > 30:
    return 1e-3
  if e > 15:
    return 1e-2
  return 1e-1

lrate = LearningRateScheduler(lrf)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.1,
                              patience=5, 
                              verbose=1,
                              min_lr=1e-5)

print('Using real-time data augmentation.')

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)

x_test = np.array([datagen.standardize(x) for x in x_test])

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.

y_test_pred_onehot = model.predict(x_test)
y_test_pred = np.argmax(y_test_pred_onehot, axis=1)+1


lines_of_text = ["{0},{1}\n".format(x[0]+1,x[1]) for x in zip(range(len(y_test_pred)), y_test_pred)]



fh=open("output.csv","w")
fh.writelines(["id,Class\n"])
fh.writelines(lines_of_text)
fh.close()

