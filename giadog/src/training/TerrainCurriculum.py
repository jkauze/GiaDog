"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
# Utils
import numpy as np
from typing import List
from utils import Particle
from random import randint, choice, choices, uniform

# Train
from GiadogEnv import GiadogEnv
from ARSHandler import ARSHandler
from PPOHandler import PPOHandler
from TRPOHandler import TRPOHandler
from TestHandler import TestHandler
from TrainHandler import TrainHandler
from __env__ import N_TRAJ, P_REPLAY, N_EVALUATE, N_PARTICLES, \
    P_TRANSITION, MIN_DESIRED_TRAV, MAX_DESIRED_TRAV, RANDOM_STEP_PROP, \
    HILLS_RANGE, STEPS_RANGE, STAIRS_RANGE

RANGES = {
    'hills'  : HILLS_RANGE,
    'steps'  : STEPS_RANGE,
    'stairs' : STAIRS_RANGE
}
INIT_VALUES = {
    'hills'  : {
        "roughness" : HILLS_RANGE["roughness"][0], 
        "frequency" : HILLS_RANGE["frequency"][0], 
        "amplitude" : HILLS_RANGE["amplitude"][0],
    },
    'steps'  : {
        "width"  : STEPS_RANGE["width"][1], 
        "height" : STEPS_RANGE["height"][0], 
    },
    'stairs' : {
        "width"  : STAIRS_RANGE["width"][1], 
        "height" : STAIRS_RANGE["height"][0], 
    }
}
METHODS = {
    'TRPO' : TRPOHandler,
    'PPO'  : PPOHandler,
    'ARS'  : ARSHandler,
    'Test' : TestHandler
}

class TerrainCurriculum(object):
    """
        [TODO]
    """
    def __init__(
            self, 
            gym_envs: List[GiadogEnv], 
            train_method: str,
            _continue: bool,
            *args,
            **kwargs
        ):
        """
            [TODO]
        """
        assert len(gym_envs) > 0, 'Must be one or more gym environments.'
        self.gym_envs = gym_envs
        self.train_method = train_method

        hills_C_t  = [
            Particle('hills', INIT_VALUES['hills'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        steps_C_t  = [
            Particle('steps', INIT_VALUES['steps'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        stairs_C_t = [
            Particle('stairs', INIT_VALUES['stairs'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        self.C_t = hills_C_t + steps_C_t + stairs_C_t
        self.C_t_history  = [[p.copy() for p in self.C_t]]

        self.train_method : TrainHandler = METHODS[train_method](
            gym_envs, 
            _continue,
            *args,
            **kwargs
        )

    def __compute_measurement_probs(self):
        """
            [TODO]
        """
        for p in self.C_t: 
            p.measurement_prob = sum(
                int(MIN_DESIRED_TRAV <= t <= MAX_DESIRED_TRAV)
                for t in p.traverability
            ) / (N_TRAJ * N_EVALUATE)

    def __update_weights(self):
        """
            [TODO]
        """
        hills_prob_sum = sum(p.measurement_prob for p in self.C_t if p.type == 'hills')
        steps_prob_sum = sum(p.measurement_prob for p in self.C_t if p.type == 'steps')
        stairs_prob_sum = sum(p.measurement_prob for p in self.C_t if p.type == 'stairs')

        for p in self.C_t: 
            if p.type == 'hills': prob_sum = hills_prob_sum
            if p.type == 'steps': prob_sum = steps_prob_sum
            if p.type == 'stairs': prob_sum = stairs_prob_sum

            if prob_sum != 0: p.weight = p.measurement_prob / prob_sum
            else:  p.weight = 1 / N_PARTICLES

    def __resample(self):
        """
            [TODO]
        """
        hills_C_t = choices(
            [p for p in self.C_t if p.type == 'hills'], 
            [p.measurement_prob for p in self.C_t if p.type == 'hills'], 
            k=N_PARTICLES
        )
        steps_C_t = choices(
            [p for p in self.C_t if p.type == 'steps'], 
            [p.measurement_prob for p in self.C_t if p.type == 'steps'], 
            k=N_PARTICLES
        )
        stairs_C_t = choices(
            [p for p in self.C_t if p.type == 'stairs'], 
            [p.measurement_prob for p in self.C_t if p.type == 'stairs'], 
            k=N_PARTICLES
        )
        self.C_t = [p.copy() for p in hills_C_t + steps_C_t + stairs_C_t]

    def __random_walk(self, p: Particle) -> Particle:
        """
            [TODO]
        """
        for attr in p.parameters:
            step = (RANGES[p.type][attr][1] - RANGES[p.type][attr][0]) * RANDOM_STEP_PROP

            p.parameters[attr] += uniform(-step, step)
            p.parameters[attr] = np.clip(
                p.parameters[attr],
                RANGES[p.type][attr][0],
                RANGES[p.type][attr][1]
            )
        return p

    def train(self):
        """
            [TODO]
        """
        epoch = 0
        while True:
            for k in range(N_EVALUATE):
                # We add the tasks for the trajectory generators
                for p in self.C_t:
                    for m in range(N_TRAJ):
                        self.train_method.gen_trajectory(p, k, m)

                self.C_t = self.train_method.extract_results(len(self.C_t))
            
            # Compute measurement probabilities
            self.__compute_measurement_probs()

            # Update weights w_j
            self.__update_weights()

            # Resample N_particle parameters
            self.__resample()

            # Append parameters to the replay memory
            self.C_t_history.append([p.copy() for p in self.C_t])

            # Create new generation
            for i in range(len(self.C_t)):
                prob = randint(0, 100) / 100

                # By p_replay probability, sample from replay memory
                if prob <= P_REPLAY:
                    while True:
                        new_p = choice(choice(self.C_t_history))
                        if new_p.type == self.C_t[i].type: break
                    self.C_t[i] = new_p

                # By ptransition probability, move C_T to an adjacent value in C.
                elif P_REPLAY < prob <= P_REPLAY + P_TRANSITION:
                    self.C_t[i] = self.__random_walk(self.C_t[i])

            epoch += 1
