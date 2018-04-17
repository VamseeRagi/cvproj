from IPython import embed
import keras
import numpy as np
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Add, merge, Lambda
from keras import regularizers
from keras.models import Sequential, Model


vgg = VGG16(weights='imagenet', include_top=False, input_shape=(48,48, 3))

def my_init(initval):
  def custominit(shape, dtype=None):
    return initval
  return custominit
  
# opt: 
# opt["model"] = {"vgg", "wide_resnet"}
# opt["fc_after_cnn"] = {True,False}
# opt["fc_after_cnn_dim"] = _Integer_
# opt["comparing_method"] = {"dot","mmul"}
# opt["comparing_method_mmul_w_dim"] = _Integer_
# opt["comparing_method_mmul_use_original_target"] = {True,False}

def build_model(shape, w_reg, k, opt={}):
  # shape should be (k+1)x100x100x3 for some k
  # first is target, rest are candidates
  
  dim_x, dim_y, dim_depth = shape[1],shape[2],shape[3]
  
  input = Input(shape=shape)
  # hide k+1 headshots in one batch to tie weights
  x = Lambda(lambda i: K.tf.reshape(i, (-1, dim_x,dim_y,dim_depth)))(input)
  
  cnn = None
  if opt["model"] == "vgg":
    cnn = vgg_f
  elif opt["model"] == "wide_resnet":
    cnn = wide_resnet_f
  cnn_out = cnn(x, w_reg)
  
  # Flatten CNN
  flattened = Lambda(lambda i: K.tf.reshape(i, (-1, i.shape[1]*i.shape[2]*i.shape[3])))(cnn_out)
  x = flattened
  
  print("CNN output size:", x.shape[1].value)
  
  if "fc_after_cnn" in opt and opt["fc_after_cnn"]:
    x = Dense(opt["fc_after_cnn_dim"], 
            activation="relu",
            kernel_initializer="glorot_normal", 
            kernel_regularizer=regularizers.l2(w_reg))(x)
    x = BatchNormalization()(x)
  dims = x.shape[1].value
  
  print("Embedding size:", dims)
  
  x = Lambda(lambda i: K.tf.reshape(i, [-1, k+1, dims]))(x)

  target = x[:,0,:] # [n, dims]
  candidates = x[:,1:,:] # [n, k, dims]  
  if opt["comparing_method"] == "dot":
    print("Doing dot product similarity")
    
    _x = K.tf.multiply(candidates, K.tf.reshape(target, [-1,1,dims]))
    _x = K.tf.reshape(_x, [-1, dims])
    x = Lambda(lambda i: _x)(x)
    x = Dense(1, kernel_initializer="glorot_normal", kernel_regularizer=regularizers.l2(w_reg))(x)
    x = Lambda(lambda i: K.tf.reshape(i, [-1, k]))(x)
    x = Activation("softmax")(x)
  elif opt["comparing_method"] == "mmul":
    print("Doing mmul weights prediction with shape:", [dims, opt["comparing_method_mmul_w_dim"]])
    
    # use original vs slimmed embedding
    if "comparing_method_mmul_use_original_target" in opt and opt["comparing_method_mmul_use_original_target"]:
      _c = Lambda(lambda i: K.tf.reshape(flattened, [-1,k+1, flattened.shape[1].value])[:,0,:])(x)
    else:
      _c = Lambda(lambda i: target)(x)
      
    c = Dense(dims * opt["comparing_method_mmul_w_dim"], 
            activation="relu",
            kernel_initializer="glorot_normal", 
            kernel_regularizer=regularizers.l2(w_reg))(_c)
    
    _w = K.tf.reshape(c, [-1, dims, opt["comparing_method_mmul_w_dim"]])
    _x = K.tf.reshape(K.tf.matmul(candidates, _w), [-1, k, opt["comparing_method_mmul_w_dim"]])
    _x = K.tf.reshape(_x, [-1, opt["comparing_method_mmul_w_dim"]])
    x = Lambda(lambda i: _x)(c)
    
    x = Dense(1, kernel_initializer="glorot_normal", kernel_regularizer=regularizers.l2(w_reg))(x)
    x = Lambda(lambda i: K.tf.reshape(i, [-1, k]))(x)
    x = Activation("softmax")(x)
  
  return Model(inputs=input, outputs=x)
  

def vgg_f(x, w_reg):
  # first 3 groups of vgg
  x = Conv2D(64, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[1].weights[0]),
             bias_initializer=my_init(vgg.layers[1].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = Conv2D(64, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[2].weights[0]),
             bias_initializer=my_init(vgg.layers[2].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(2)(x)
  
  x = Conv2D(128, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[4].weights[0]),
             bias_initializer=my_init(vgg.layers[4].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = Conv2D(128, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[5].weights[0]),
             bias_initializer=my_init(vgg.layers[5].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(2)(x)
  
  x = Conv2D(256, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[7].weights[0]),
             bias_initializer=my_init(vgg.layers[7].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = Conv2D(256, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[8].weights[0]),
             bias_initializer=my_init(vgg.layers[8].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = Conv2D(256, 3, padding="same", activation="relu", 
             kernel_initializer=my_init(vgg.layers[9].weights[0]),
             bias_initializer=my_init(vgg.layers[9].weights[1]),
             kernel_regularizer=regularizers.l2(w_reg))(x)
  x = BatchNormalization()(x)
  x = MaxPooling2D(2)(x)
  
  return x

  
def wide_resnet_f(x, w_reg):
  n = 2 # depth = 6*n + 4
  k = 4 # widen factor
  
  def bottleneck(x, out_filter, dropout=None, first_stride=2):
    for i in range(n):
      print(i,n, x.shape)
      shortcut = x
      
      # 1st conv
      x = Activation("relu")(BatchNormalization()(x))
      # only use shortcut if not bottleneck in a group
      if i == 0:
        shortcut = x
      # decrease dimension on very first conv
      x = Conv2D(out_filter, 3, strides=(first_stride if i == 0 else 1), 
                 padding="same", kernel_initializer="glorot_normal",
                 kernel_regularizer=regularizers.l2(w_reg))(x)
      
      # 2nd conv           
      x = Activation("relu")(BatchNormalization()(x))
      if dropout is not None:
        x = Dropout(dropout)(x)
      x = Conv2D(out_filter, 3, strides=1, 
                 padding="same", kernel_initializer="glorot_normal",
                 kernel_regularizer=regularizers.l2(w_reg))(x)
      
      if i == 0:
        shortcut = Conv2D(out_filter, 1, strides=first_stride, 
                          padding="same", kernel_initializer="glorot_normal",
                          kernel_regularizer=regularizers.l2(w_reg))(shortcut)
      x = Add()([shortcut, x])
      #x = merge([shortcut, x], mode="sum")
    return x
  
  x = Conv2D(16, 3, padding="same", kernel_initializer="glorot_normal",
             kernel_regularizer=regularizers.l2(w_reg))(x)
  
  x = bottleneck(x, 16*k, dropout=None, first_stride=1)
  x = bottleneck(x, 32*k, dropout=None)
  x = bottleneck(x, 64*k, dropout=None)
  x = Activation("relu")(BatchNormalization()(x))
  x = AveragePooling2D((8, 8), strides=(1, 1))(x)
  #x = Flatten()(x)
  # x = Dense(num_classes, 
            # activation="softmax",
            # kernel_initializer="glorot_normal", 
            # kernel_regularizer=regularizers.l2(w_reg))(x)

  return x#Model(inputs=input, outputs=x) 