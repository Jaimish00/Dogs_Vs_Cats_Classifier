import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from datetime import datetime

# gpu_options = tf.GPUOptions(per_process_gpu_memory_function=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

# Normlize Data
X = X/255.0

# Best ----> 3 Convs, 64 Nodes, 0 Dense
# dense_layers = [0, 1, 2]
# layers_size = [32, 64, 128]
# conv_layers = [1, 2, 3]

dense_layers = [0]
layers_size = [64]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layers_size:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-"+datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
            tensorboard = tf.keras.callbacks.TensorBoard(log_dir=f'logs/{NAME}')
            print(NAME)
            model = Sequential()

            model.add(Conv2D(layer_size, (3,3), input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for l in range(dense_layer):
                model.add(Dense(512))
                model.add(Activation("relu"))

            # model.add(Dropout(0.8))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                         optimizer="adam",
                         metrics=["accuracy"])

            model.fit(X, y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[tensorboard])
