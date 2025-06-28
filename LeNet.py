import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import numpy as np
import os

def build_lenet():
    #Whole architecture of the lenet
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6,(5,5),activation='tanh',input_shape=(32,32,1),padding = 'same'),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Conv2D(16,(5,5),activation='tanh'),
        tf.keras.layers.AveragePooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120,activation = 'tanh'),
        tf.keras.layers.Dense(84,activation = 'tanh'),
        tf.keras.layers.Dense(10,activation = 'softmax')

    ])
    return model