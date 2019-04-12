import keras
from keras.models import Sequential, load_model, Model
from keras.layers import (Dense,
                          Activation,
                          Dropout, Conv2D,
                          ZeroPadding2D,
                          MaxPooling2D,
                          Flatten,
                          PReLU,
                          BatchNormalization,
                          Input,
                          Add,
                          Multiply,
                          AveragePooling2D)

from keras.regularizers import l2
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.initializers import glorot_uniform

from utils import PARAM

def model_cnn():   
    filter_pixel = 3
    input_shape = PARAM.input_shape
    
    model = Sequential()
    model.add(Conv2D(256, kernel_size=(filter_pixel, filter_pixel), padding="same",
                 activation='relu',
                 input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(512, kernel_size=(2,2), activation='relu',padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Conv2D(256, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    
    model.add(Conv2D(256, kernel_size=(filter_pixel, filter_pixel), activation='relu',padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    model.add(Flatten())
    
    for i in range(2):
        model.add(Dense(256))
        model.add(PReLU())
        model.add(Dropout(0.5))

    model.add(Dense(7, activation='softmax'))
    return model

#ResNet Block
def resnet_block(inputs,num_filters=16,
                  kernel_size=3,strides=1,
                  activation='relu'):
    
    x = Conv2D(num_filters,
               kernel_size=kernel_size,
               strides=strides,padding='same',
               kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    
    if(activation):
        x = PReLU()(x)
        
    return x

def model_resnet(input_shape=PARAM.input_shape):
    inputs = Input(shape=input_shape)
    
    #first layer
    x = resnet_block(inputs)
    for i in range(4):
        a = resnet_block(inputs = x)
        b = resnet_block(inputs=a,activation=None)
        x = keras.layers.add([x,b])
        x = PReLU()(x)
        
    for i in range(6):
        if i == 0:
            a = resnet_block(inputs = x,strides=2,num_filters=32)
        else:
            a = resnet_block(inputs = x,num_filters=32)
        b = resnet_block(inputs=a,activation=None,num_filters=32)
        if i==0:
            x = Conv2D(32,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
            
        x = keras.layers.add([x,b])
        x = PReLU()(x)

    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x,strides=2,num_filters=64)
        else:
            a = resnet_block(inputs = x,num_filters=64)

        b = resnet_block(inputs=a,activation=None,num_filters=64)
        if i == 0:
            x = Conv2D(64,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
            
        x = keras.layers.add([x,b])
        x = PReLU()(x)
        
    for i in range(6):
        if i ==0 :
            a = resnet_block(inputs = x, strides=2,num_filters=128)
        else:
            a = resnet_block(inputs = x,num_filters=128)

        b = resnet_block(inputs=a,activation=None,num_filters=128)
        if i == 0:
            x = Conv2D(128,kernel_size=3,strides=2,padding='same',
                       kernel_initializer='he_normal',kernel_regularizer=l2(1e-4))(x)
            
        x = keras.layers.add([x,b])
        x = PReLU()(x)
    
    att = Dense(int(x.shape[3]), kernel_initializer=glorot_uniform(), activation='softmax')(x)
    x = Multiply()([att, x])

    x = AveragePooling2D(pool_size=2)(x)
    y = Flatten()(x)
    outputs = Dense(7,activation='softmax',
                    kernel_initializer='he_normal')(y)
    
    model = Model(inputs=inputs,outputs=outputs)
    return model