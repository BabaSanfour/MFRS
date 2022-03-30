import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout


def deepid_architecture(num_classes: int =1000, in_shape=(55,47,3)):
    myInput = Input(shape=in_shape)

    x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=in_shape)(myInput)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
    x = Dropout(rate=0.99, name='D1')(x)

    x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
    x = Dropout(rate=0.99, name='D2')(x)

    x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
    x = Dropout(rate=0.99, name='D3')(x)

    x1 = Flatten()(x)
    fc11 = Dense(160, name = 'fc11')(x1)

    x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name = 'fc12')(x2)

    y = Add()([fc11, fc12])
    y = Activation('relu', name = 'deepid')(y)
    y = Dense(num_classes, activation='softmax', name='classifier')(y)

    model = Model(inputs=[myInput], outputs=y)
    return model


def DeepID(pretrained: bool = False, num_classes: int =1000):

    model = deepid_architecture(160, (55,47,3))
    if pretrained:
        model.load_weights('/home/hamza97/scratch/net_weights/deepid_keras_weights.h5')
    deepid_model = deepid_architecture( num_classes, (224,224,1))
    for i in range(2,13):
        deepid_model.layers[i].set_weights(model.layers[i].get_weights())
    return deepid_model
