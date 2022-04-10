"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]

"""
import numpy as np
from ANN import ANN
from typing import *
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from __env__ import PRIVILIGED_DATA, NON_PRIVILIGED_DATA, \
    PRIVILIGED_DATA_SHAPE, NON_PRIVILIGED_DATA_SHAPE

class TeacherValueANN(ANN):
    """ Teacher value ANN class. """
    
    def __init__(self, observation_space: spaces.Dict):
        """
            Initializes the teacher value ANN.

            Arguments:
            ----------
                observation_space: gym.spaces.Dict
                    The observation space of the environment
        """
        self.observation_space = observation_space

        inputs_x_t = keras.Input(
            shape=PRIVILIGED_DATA_SHAPE, 
            name='priviliged_data'
        )

        inputs_o_t = keras.Input(
            shape=NON_PRIVILIGED_DATA_SHAPE,
            name='non_priviliged_data'
        )

        concat = keras.layers.Concatenate()([inputs_x_t, inputs_o_t])
        
        x = keras.layers.Dense(256, activation='tanh')(concat)
        x = keras.layers.Dense(128, activation='tanh')(x)
        x = keras.layers.Dense(64, activation='tanh')(x)
        x = keras.layers.Dense(16, activation='tanh')(x)
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(
            inputs= [inputs_x_t, inputs_o_t], 
            outputs=outputs, 
            name='value ANN'
        )

    def __format_states(
            self, 
            states: List[Dict[str, np.array]]
        ) -> Tuple[np.array, np.array]:
        """
            Get dictionary states and convert them to numpy arrays.
        """
        input_x_t = np.array([
            np.concatenate([np.reshape(s[d], -1) for d in PRIVILIGED_DATA])
            for s in states
        ])

        input_o_t = np.array([
            np.concatenate([np.reshape(s[d], -1) for d in NON_PRIVILIGED_DATA])
            for s in states
        ])

        return input_x_t, input_o_t
    
    def __call__(self, states: Dict[str, np.array]) -> tf.Tensor:
        """
            Computes the value of the states.

            Arguments:
            ----------
                states: Dict[str, numpy.array]
                    States to process

            Return:
            -------
                tensorf.Tensor
                    Value computed
        """
        # Get ANN input
        assert self.verify_states(states)
        input_x_t, input_o_t = self.__format_states(states)

        return self.model([input_x_t, input_o_t])

class StudentValueANN(ANN):
    pass
