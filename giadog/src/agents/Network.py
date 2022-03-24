import numpy as np
from typing import *


class Network(object):
    def __init__(self): raise NotImplementedError

    def __call__(self): raise NotImplementedError

    def verify_states(self, states: List[Dict[str, np.array]]) -> bool:
        """
            Verify that all states are part of the observation space
        """
        return all(self.observation_space.contains(s) for s in states)


    def save(self, path: str):
        """
            Saves the model weights to a directory

            Arguments:
            ----------
                path: str 
                    Path where the model will be saved.
        """
        self.model.save_weights(path)

    def load(self, path: str):
        """
            Loads the model weights from a directory.

            Arguments:
            ----------
                path: str 
                    Path to the file where the weights was loaded.
        """
        self.model.load_weights(path)