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


# train_mat = loadmat("train.mat.txt")
# test_mat = loadmat("test.mat.txt")

# def bar(x):
  # xb = x.copy()
  # for i in range(len(xb)):
    # xb[i,:,0:4,0] = int(np.mean(xb[i][:,:,0]))
    # xb[i,:,0:4,1] = int(np.mean(xb[i][:,:,0]))
    # xb[i,:,0:4,2] = int(np.mean(xb[i][:,:,0]))
    # xb[i,:,28:32,0] = int(np.mean(xb[i][:,:,0]))
    # xb[i,:,28:32,1] = int(np.mean(xb[i][:,:,1]))
    # xb[i,:,28:32,2] = int(np.mean(xb[i][:,:,2]))
  # return xb

# x_train = bar(train_mat['X'].transpose([3,0,1,2]))
# y_train = train_mat['y'] - 1

# valid_mat = loadmat("test_32x32.mat")
# x_valid = bar(valid_mat['X'].transpose([3,0,1,2]))

# y_valid = valid_mat['y'] - 1


  
# x_test = bar(test_mat['X'].transpose([3,0,1,2]))


batch_size = 100
num_classes = 10
epochs = 100
data_augmentation = True
#num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'data2.h5'

# The data, split between train and test sets:
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

w_reg = 1e-4

#%%
model = build_model([11,100,100,3], 1e-4, 10, opt={
  "model": "vgg", 
  "fc_after_cnn": True,
  "fc_after_cnn_dim": 200,
  "comparing_method": "mmul",
  "comparing_method_mmul_w_dim": 20,
  "comparing_method_mmul_use_original_target": True,
  "": ""
})
#%%

opt = keras.optimizers.SGD(lr=0.01, decay=0.0, momentum=0.90, nesterov=False)
#opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_test /= 255
x_valid /= 255


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
                              
# if not data_augmentation:
    # print('Not using data augmentation.')
    # history = model.fit(x_train, y_train,
              # batch_size=batch_size,
              # epochs=epochs,
              # validation_data=(x_valid, y_valid),
              # shuffle=True)
# else:
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
x_valid = np.array([datagen.standardize(x) for x in x_valid])


def build_data_map(path):
    print("") #placeholder so we don't have errors when running
  # TODO
  # make it so its one label per folder
  

def my_data_generator(batch_size):
  # consult https://github.com/keras-team/keras/issues/8078
  while True:
    # ends up being [batchsize, 10+1, 100, 100, 3] 
    # [:,0,:,:,:] is target, [:,1:,:,:,:] is candidates
    batch_imgs = [] 
    
    #ends up being [batchsize, 10] (onehot encoded)
    batch_labels = [] 
    
    for i in range(batch_size):
        print("") #placeholder so we don't have errros when running
      # TODO and append to lists
      # pick 10 random candidates, then pick 1 of the ten as target, and grab 11 images
      # make img and label
    
    yield np.array(batch_imgs), np.array(batch_labels)
      
# TODO replace the fit_generator call with equivalent that generates random batches
# Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(batch_size,
                    epochs=epochs,
                    validation_data=(x_valid, y_valid),
                    callbacks=[lrate],
                    #validation_split = 0.02,
                    workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.

y_test_pred_onehot = model.predict(x_test)
y_test_pred = np.argmax(y_test_pred_onehot, axis=1)+1
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])

#print(history.history['val_acc'])


lines_of_text = ["{0},{1}\n".format(x[0]+1,x[1]) for x in zip(range(len(y_test_pred)), y_test_pred)]

#[print(x) for x in lines_of_text]
# Write prediction to file

fh=open("output.csv","w")
fh.writelines(["id,Class\n"])
fh.writelines(lines_of_text)
fh.close()

