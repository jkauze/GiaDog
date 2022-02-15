"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
# Utilities
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
    NON_PRIVILIGED_DATA_SHAPE = 121
    PRIVILIGED_DATA_SHAPE     = 59
    CLASSIFIER_INPUT_SHAPE    = 64 + NORMAL_DATA_SHAPE 

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
        inputs_x_t = keras.Input(shape=self.PRIVILIGED_DATA_SHAPE)
        x_t = layers.Dense(72, activation='tanh')(inputs_x_t)
        x_t = layers.Dense(64, activation='tanh')(x_t)
        self.encoder = keras.Model(inputs_x_t, x_t)

        # Concatenate the output of the previous network with the non-privileged data
        inputs_o_t = keras.Input(shape=self.NON_PRIVILIGED_DATA_SHAPE)
        concat = layers.Concatenate()([inputs_o_t, self.encoder])

        # Classifier subnetwork
        inputs_c = keras.Input(shape=self.CLASSIFIER_INPUT_SHAPE)
        x = layers.Dense(256, activation='tanh')(inputs_c)
        x = layers.Dense(128, activation='tanh')(x)
        x = layers.Dense(64, activation='tanh')(x)
        outputs_c = layers.Dense(16)(x)
        self.classifier = keras.Model(inputs_c, outputs_c)

        # Entire network
        outputs = self.classifier(concat)
        self.model = keras.Model([inputs_x_t, inputs_o_t], outputs)

    def predict(self, input_x_t, input_o_t) -> np.array:
        return self.model.predict([input_x_t, input_o_t])

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
        inputs_h_t = keras.Input(shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE))
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE),
            nb_filters=self.NORMAL_DATA_SHAPE,
            kernel_size=5,
            dilatations=(1,),
            activation='relu',
            return_sequences=True
        )(inputs_h_t)
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE),
            nb_filters=self.NORMAL_DATA_SHAPE // 2,
            kernel_size=5,
            dilatations=(1,),
            activation='relu',
            return_sequences=True
        )(h_t)
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE // 2),
            nb_filters=self.NORMAL_DATA_SHAPE // 2,
            kernel_size=5,
            dilatations=(2,),
            activation='relu',
            return_sequences=True
        )(h_t)
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE // 2),
            nb_filters=self.NORMAL_DATA_SHAPE // 4,
            kernel_size=5,
            dilatations=(2,),
            activation='relu',
            return_sequences=True
        )(h_t)
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE // 4),
            nb_filters=self.NORMAL_DATA_SHAPE // 4,
            kernel_size=5,
            dilatations=(4,),
            activation='relu',
            return_sequences=True
        )(h_t)
        h_t = TCN(
            input_shape=(None, HISTORY_LEN, self.NORMAL_DATA_SHAPE // 4),
            nb_filters=self.NORMAL_DATA_SHAPE // 8,
            kernel_size=5,
            dilatations=(2,),
            activation='relu',
            return_sequences=False
        )(h_t)
        h_t = layers.Dense(64, activation='tanh')(h_t)
        self.encoder = keras.Model(inputs_h_t, h_t)

        # Concatenate the output of the previous network with the non-privileged data
        inputs_o_t = keras.Input(shape=self.NON_PRIVILIGED_DATA_SHAPE)
        concat = layers.Concatenate()([inputs_o_t, self.encoder])

        # Classifier subnetwork inherited from teacher
        self.classifier_model = teacher.classifier_model

        # Entire network
        outputs = self.classifier_model(concat)
        self.model = keras.Model([inputs_h_t, inputs_o_t], outputs)




