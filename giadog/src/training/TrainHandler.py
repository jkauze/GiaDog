from typing import List
from utils import Particle
from abc import abstractmethod
from GiadogEnv import GiadogEnv

class TrainHandler(object):
    """
        [TODO]
    """
    @abstractmethod
    def __init__(self, envs: List[GiadogEnv], _continue: bool): pass 

    @abstractmethod
    def gen_trajectory(self, p: Particle, k: int, m: int): pass

    @abstractmethod
    def extract_results(self) -> List[Particle]: pass