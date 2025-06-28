import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os

def residual_block(x, filters):
    shortcut = x

    # First conv layer
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    # Second conv layer
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)

    # Match dimensions if needed using 1x1 conv
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    # Add and activate
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x



def build_resnet():
    input = tf.keras.Input((32,32,3))
    x = layers.Conv2D(64,(3,3),padding ='same',activation='relu')(input)
    x = residual_block(x,64)
    x = layers.MaxPooling2D((2,2))(x)
    x = residual_block(x,128)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(10,activation='softmax')(x)
    model = models.Model(input,output)
    return model





