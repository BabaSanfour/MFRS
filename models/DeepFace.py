import os

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Convolution2D, LocallyConnected2D, MaxPooling2D, Flatten, Dense, Dropout


def deepface_architecture(num_classes: int =1000, in_shape=(152,152,3)):
    base_model = Sequential()
    base_model.add(Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=in_shape))

    base_model.add(MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
    base_model.add(Convolution2D(16, (9, 9), activation='relu', name='C3'))
    base_model.add(LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
    base_model.add(LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
    base_model.add(LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
    base_model.add(Flatten(name='F0'))
    base_model.add(Dense(4096, activation='relu', name='F7'))
    base_model.add(Dropout(rate=0.5, name='D0'))
    base_model.add(Dense(num_classes, activation='softmax', name='F8'))
    return base_model


def DeepFace(pretrained: bool = False, num_classes: int =1000):

    model = deepface_architecture(8631, (152,152,3))
    if pretrained:
        model.load_weights('/home/hamza97/scratch/net_weights/VGGFace2_DeepFace_weights_val-0.9034.h5')
    deepface_model = deepface_architecture( num_classes, (152,152,1))
    for i in range(1,8):
        deepface_model.layers[i].set_weights(model.layers[i].get_weights())
    deepface_model=tf.keras.models.Model(inputs=deepface_model.layers[0].input, outputs=deepface_model.layers[-1].output)
    return deepface_model
