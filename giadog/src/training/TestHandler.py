"""
    [TODO]
"""
# Utilities
import numpy as np
from uuid import uuid4
from typing import List
from random import randint
from utils import Particle

# Train
from GiadogEnv import GiadogEnv
from multiprocessing import Queue
from TrainHandler import TrainHandler
from __env__ import N_TRAJ, MAX_DESIRED_TRAV, MIN_DESIRED_TRAV, \
    HILLS_RANGE, STEPS_RANGE, STAIRS_RANGE, ROWS, COLS


class TestHandler(TrainHandler):
    """
        [TODO]
    """

    def __init__(
            self, 
            envs: List[GiadogEnv], 
            _continue: bool, 
            type: str,
            epoch_to_show: int,
            roughness: float,
            frequency: float,
            amplitude: float,
            width: float, 
            height: float
        ):
        self.env = envs[0]
        self.type = type 
        self.epoch_to_show = epoch_to_show
        self.roughness = roughness
        self.frequency = frequency
        self.amplitude = amplitude
        self.width = width 
        self.height = height
        self.epoch = 0
        self.results = Queue()

    def gen_trajectory(
            self,
            env: GiadogEnv,
            p: Particle, 
            k: int, 
            m: int, 
            results: Queue,
        ):
        if p.type != self.type: return 

        mean = (MAX_DESIRED_TRAV + MIN_DESIRED_TRAV) / 2
        std = 0.0625

        if p.type == 'hills':
            fitness = abs(self.roughness - p.parameters['roughness']) / \
                (HILLS_RANGE['roughness'][1] - HILLS_RANGE['roughness'][0])
            fitness += abs(self.frequency - p.parameters['frequency']) / \
                (HILLS_RANGE['frequency'][1] - HILLS_RANGE['frequency'][0])
            fitness += abs(self.amplitude - p.parameters['amplitude']) / \
                (HILLS_RANGE['amplitude'][1] - HILLS_RANGE['amplitude'][0])
            fitness /= 3
        elif p.type == 'steps':
            fitness = abs(self.width - p.parameters['width']) / \
                (STEPS_RANGE['width'][1] - STEPS_RANGE['width'][0])
            fitness += abs(self.height - p.parameters['height']) / \
                (STEPS_RANGE['height'][1] - STEPS_RANGE['height'][0])
            fitness /= 2
        else:
            fitness = abs(p.parameters['width'] - self.width) / \
                (STAIRS_RANGE['width'][1] - STAIRS_RANGE['width'][0])
            fitness += abs(self.height - p.parameters['height']) / \
                (STAIRS_RANGE['height'][1] - STAIRS_RANGE['height'][0])
            fitness /= 2

        mean += fitness
        p.traverability[k * N_TRAJ + m] = np.clip(np.random.normal(mean, std), 0, 1)

        # Send results
        results.put(p)

    def extract_results(self, N: int) -> List[Particle]:
        """
            [TODO]
        """
        # Get data from all trajectories
        C_t = [self.results.get() for _ in range(N * N_TRAJ)]

        # Show the terrain with the best fitness
        best_p = None 
        best_fitness = -1 
        for p in [p for p in C_t if p.type == self.type]:
            if p.weight > best_fitness:
                best_p = p 
                best_fitness = p.weight
        
        if self.env.sim.gui and self.epoch % self.epoch_to_show == 0:
            terrain_file = f'terrains/{best_p.type}_{uuid4()}.txt'
            self.env.make_terrain(
                terrain_file,
                best_p.type,
                rows=ROWS,
                cols=ROWS,
                seed=randint(0, 1e6),
                **best_p.parameters
            )
            self.env.reset(terrain_file)

        parameters = {
            attr: '{:.4f}'.format(best_p.parameters[attr]) \
            for attr in best_p.parameters
        }
        T = len(best_p.traverability)
        traverability = '{:.4f}'.format(sum(best_p.traverability) / T)
        print(
            f'\033[1;36m[i]\033[0m Epoch: {self.epoch} |' +\
            f'Best parameters: {parameters} | ' +\
            f'Traverability : {traverability} | '
            'Weight: {:.4f}'.format(best_p.weight)
        )

        self.epoch += 1