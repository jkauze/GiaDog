# Utilities
import json
from typing import *
from abc import abstractmethod

# Machine Learning
import numpy as np
from tcn import TCN
from tensorflow import keras, clip_by_value, Variable
from tensorflow.keras import layers
from src.distributions import make_dist


from gym import spaces


class teacher_network(object):
    """
        Teacher policy network class.

    """
    def __init__(self, 
                action_space:Optional[spaces.Box]=None,
                observation_space:Optional[spaces.Box]=None):
        """
        Initializes the teacher network.

        Arguments:
        ----------
        action_space: gym.spaces.Box -- The action space of the environment.

        Returns:
        --------
        None
        """
        self.PRIVILIGED_DATA_SHAPE = 59
        self.NON_PRIVILIGED_DATA_SHAPE = 145
        self.CLASSIFIER_INPUT_SHAPE = 64 + self.NON_PRIVILIGED_DATA_SHAPE

    
        # Subnetwork that processes the privileged data
        inputs_x_t = keras.Input(
            shape=self.PRIVILIGED_DATA_SHAPE, 
            name='priviliged_data'
        )
        x_t = layers.Dense(72, activation='tanh')(inputs_x_t)
        x_t = layers.Dense(64, activation='tanh')(x_t)
        self.encoder = keras.Model(inputs_x_t, x_t, name='encoder')

        # Concatenate the output of the previous network with the non-privileged 
        # data
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
        mean = self.classifier(concat)

       

        # Other variables
        # Get the policy distribution
        self.policy_dist = make_dist(action_space)

        self.log_std = Variable(np.zeros(self.policy_dist.ndim, 
                                                dtype=np.float32))
        
        # Log std (for the Gaussian distribution) 
        
        self.model = keras.Model(inputs = [inputs_x_t, inputs_o_t], 
                                outputs = mean)

        # Get the action space and the observation space
        self.action_space = action_space
        self.observation_space = observation_space

        

    def predict(self, input_x_t, input_o_t) -> np.array:
        return self.model([input_x_t, input_o_t])
    
    def __call__(self, input, greedy = False):

        input_x_t, input_o_t = input
        
        mean = self.model([input_x_t, input_o_t])

        self.policy_dist.set_param([mean, self.log_std])
        
        if greedy:
            result = self.policy_dist.greedy_sample()
        else:
            result = self.policy_dist.sample()
        
        
        """
        # Actions are normalized
        if greedy:
            result = result * self._action_scale + self._action_mean
        else:
            result, explore = result
            result = result * self._action_scale + self._action_mean + explore
        """

        result = clip_by_value(result, self.action_space.low, 
                                        self.action_space.high)

        return result

    
    def save_model_weights(self, path: str, epoch: int):
        """
            Saves the teacher model weights to a file and also saves separetly 
            the classifier model weights.The format of the files are: .ckpt

            Arguments:
            ----------
            path: str -- Path to the file where the model will be saved.
            epoch: int -- Epoch number (Is to indicate the creation of).

            Returns:
            --------
            None
        """
        self.model.save_weights(path + '_epoch_' + str(epoch) + '.ckpt')
        self.classifier.save_weights(path + '_classifier_epoch_' + str(epoch)\
         + '.ckpt')
    
    def save_model(self, path: str, epoch: int):
        """
            Saves the full teacher model to a file and also saves separetly the
            classifier model. The format of the files are: .h5

            Arguments:
            ----------
            path: str -- Path to the file where the model will be saved.
            epoch: int -- Epoch number (Is to indicate the creation of).

            Returns:
            --------
            None
        """
        self.model.save(path + '_epoch_' + str(epoch) + '.ckpt')
        self.classifier.save(path + '_classifier_epoch_' + str(epoch) + '.ckpt')
    
    def load_model(self, path:str):
        """
            Loads the model from a .h5 file.

            Note: The model that it is going to be loaded should be the same as
                    the teacher model. 

            Arguments:
            ----------
            path: str -- Path to the file where the model will be loaded.
                         The file must be a .h5 file.
        """

        self.model = keras.models.load_model(path)
    
    def load_weights(self, path:str):
        """
            Loads the weights from a .ckpt file.

            Note: The weights must be saved with the save_model_weights method.
                  And are the teacher model weights, (not the classifier 
                  weights).
            Arguments:
            ----------
            path: str -- Path to the file where the weights will be loaded.
                         The file must be a .ckpt file.
        """
        self.model.load_weights(path)
    

