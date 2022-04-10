# Utilities
from typing import *

# Machine Learning
import numpy as np
from ANN import ANN
from gym import spaces
import tensorflow as tf
from distributions import DiagGaussian
from tensorflow import keras, clip_by_value
from __env__ import PRIVILIGED_DATA, NON_PRIVILIGED_DATA, \
    PRIVILIGED_DATA_SHAPE, NON_PRIVILIGED_DATA_SHAPE, CLASSIFIER_INPUT_SHAPE


class VariableLayer(keras.layers.Layer):
    """
        Variable Layer class

        It is an auxiliary layer that it is used to create a varible layer.

        Its main purpose in this project is to create a layer that outputs the 
        log of the standard deviation of each of the actions. Thi is used to 
        implement a diagonal gaussian policy.


        For more info about gaussian policy, please refer to:
        https://spinningup.openai.com/en/latest/spinningup/rl_intro.html
        (Look for the appendix on Diagonal Gaussian Policies)

    """
    def __init__(self, units: int, *args, **kwargs):
        """
            Arguments:
            ----------
                units: int
                    Number of units of the layer.
        """
        super(VariableLayer, self).__init__(*args, **kwargs)
        self.units = units

    def build(self, input_shape: List[int]=None):
        """
            Build method of the layer.

            Arguments:
            ----------
                input_shape: List[int], optional
                    Dummy argument.
                    Default: None
        """
        self.bias = self.add_weight(
            'bias',
            shape=self.units,
            initializer='zeros',
            trainable=True
        )

    def call(self, x: tf.Tensor=None) -> tf.Tensor:
        """
            Call method of the layer.

                Arguments:
                ----------
                    x: tensorflow.Tensor, optional
                        Dummy argument.
                        Default: None

                Returns:
                --------
                    tensorflow.Tensor
                        Bias weights.
        """
        return self.bias

class TeacherANN(ANN):
    """
        Teacher policy ANN class.

    """
    def __init__(
            self, 
            action_space: spaces.Box,
            observation_space: spaces.Dict
        ):
        """
            Initializes the teacher ANN.

            Arguments:
            ----------
                action_space: gym.spaces.Box
                    The action space of the environment.

                observation_space: gym.spaces.Dict
                    The observation space of the environment
        """
        # Get the action space and the observation space
        self.action_space = action_space
        self.observation_space = observation_space

        # Get the policy distribution
        self.policy_dist = DiagGaussian(
            self.action_space.shape[0],
            # [TODO] hay que definir la media y la escala de las acciones
            np.zeros(16,),
            np.ones((16,))
        )

        # SubANN that processes the privileged data
        inputs_x_t = keras.Input(
            shape=PRIVILIGED_DATA_SHAPE, 
            name='priviliged_data'
        )
        x_t = keras.layers.Dense(72, activation='tanh')(inputs_x_t)
        x_t = keras.layers.Dense(64, activation='tanh')(x_t)
        self.encoder = keras.Model(inputs_x_t, x_t, name='encoder')

        # Concatenate the output of the previous ANN with the 
        # non-privileged data
        inputs_o_t = keras.Input(
            shape=NON_PRIVILIGED_DATA_SHAPE,
            name='non_priviliged_data')
        concat = keras.layers.Concatenate()([inputs_o_t, x_t])

        # Classifier subANN
        inputs_c = keras.Input(shape=CLASSIFIER_INPUT_SHAPE)
        x = keras.layers.Dense(256, activation='tanh')(inputs_c)
        x = keras.layers.Dense(128, activation='tanh')(x)
        x = keras.layers.Dense(64, activation='tanh')(x)
        outputs_c = keras.layers.Dense(16)(x)
        self.classifier = keras.Model(inputs_c, outputs_c, name='classifier')

        # Mean ANN
        mean = self.classifier(concat)
        # Log std (for the Gaussian distribution)
        log_std = VariableLayer(self.policy_dist.ndim)([inputs_x_t, inputs_o_t])

        # Full ANN
        self.model = keras.Model(
            inputs = [inputs_x_t, inputs_o_t], 
            outputs = [mean, log_std]
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

    def __call__(
            self, 
            states: Dict[str, np.array], 
            greedy: bool=False
        ) -> np.array:
        """
            Returns the action and the log probability of the action.

            Arguments:
            ----------
                states: Dict[str, numpy.array]
                    States to process

                greedy: bool, optional
                    If True, the greedy policy is used.
                    Default: False


            Returns:
            --------
                np.array, shape (16,)
                    The action to be taken.
        """
        # Get ANN input
        assert self.verify_states(states)
        input_x_t, input_o_t = self.__format_states(states)
        
        # Compute and set the distribution params (mean and logstd)
        self.policy_dist.set_param(*self.model([input_x_t, input_o_t]))
        
        if greedy: result = self.policy_dist.greedy_sample()
        else: result = self.policy_dist.sample()

        result = clip_by_value(
            result, 
            self.action_space.low,
            self.action_space.high
        )

        return result

class StudentANN(ANN):
    pass
