"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
# Utilities
import json
from typing import *
from abc import abstractmethod

# Machine Learning
import numpy as np
from tcn import TCN
from tensorflow import keras
from tensorflow.keras import layers


# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
HISTORY_LEN  = ENV["NEURAL_NETWORK"]["HISTORY_LEN"]


class controller_neural_network:
    """
        [TODO]
    """
    NORMAL_DATA_SHAPE         = 60
    NON_PRIVILIGED_DATA_SHAPE = 145
    PRIVILIGED_DATA_SHAPE     = 59
    CLASSIFIER_INPUT_SHAPE    = 64 + NON_PRIVILIGED_DATA_SHAPE

    @abstractmethod
    def __init__(self): pass 

    @abstractmethod 
    def __loss(self): pass

    @abstractmethod
    def train(self): pass 

    @abstractmethod
    def predict(self): pass

class teacher_nn(controller_neural_network):
    """
        [TODO]
    """
    def __init__(self):
        # Subnetwork that processes the privileged data
        inputs_x_t = keras.Input(
            shape=self.PRIVILIGED_DATA_SHAPE, 
            name='priviliged_data'
        )
        x_t = layers.Dense(72, activation='tanh')(inputs_x_t)
        x_t = layers.Dense(64, activation='tanh')(x_t)
        self.encoder = keras.Model(
            inputs_x_t, 
            x_t,
            name='encoder'
          )

        # Concatenate the output of the previous network with the non-privileged data
        inputs_o_t = keras.Input(
            shape=self.NON_PRIVILIGED_DATA_SHAPE,
            name='non_priviliged_data')
        concat = layers.Concatenate()([inputs_o_t, x_t])

        # Classifier subnetwork
        inputs_c = keras.Input(shape=self.CLASSIFIER_INPUT_SHAPE)
        x = layers.Dense(256, activation='tanh')(inputs_c)
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        outputs_c = layers.Dense(16)(x)
        self.classifier = keras.Model(inputs_c, outputs_c, name='classifier')

        # Entire network
        outputs = self.classifier(concat)
        self.model = keras.Model([inputs_x_t, inputs_o_t], outputs)

    def predict(self, input_x_t, input_o_t) -> np.array:
        
        return self.model.predict([input_o_t,input_x_t])

class student_nn(controller_neural_network):
    """
        [TODO]

        References:
        -----------
            * @misc{KerasTCN,
                author = {Philippe Remy},
                title = {Temporal Convolutional Networks for Keras},
                year = {2020},
                publisher = {GitHub},
                journal = {GitHub repository},
                howpublished = {\ url{https://github.com/philipperemy/keras-tcn}},
            }
    """
    def __init__(self, teacher: teacher_nn):
        # TCN network
        inputs_h_t = keras.Input(shape=(HISTORY_LEN, self.NORMAL_DATA_SHAPE))
        h_t = TCN(
            nb_filters=self.NORMAL_DATA_SHAPE // 2,
            kernel_size=5,
            dilations=(1,2,4,8,16),
            activation='relu',
            return_sequences=False
        )(inputs_h_t)
        h_t = layers.Dense(64, activation='tanh')(h_t)
        self.encoder = keras.Model(inputs_h_t, h_t)

        # Concatenate the output of the previous network with the non-privileged data
        inputs_o_t = keras.Input(shape=self.NON_PRIVILIGED_DATA_SHAPE)
        concat = layers.Concatenate()([inputs_o_t, h_t])

        # Classifier subnetwork inherited from teacher
        self.classifier = teacher.classifier

        # Entire network
        outputs = self.classifier(concat)
        self.model = keras.Model([inputs_h_t, inputs_o_t], outputs)
    
    def predict(self, input_h_t, input_o_t) -> np.array:
        return self.model.predict([input_h_t, input_o_t])

