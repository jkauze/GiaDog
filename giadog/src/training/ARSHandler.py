"""
    [TODO]
"""
# Utilities
from time import time
from typing import List
from utils import Particle

# Train
from agents import ARSModel
from GiadogEnv import GiadogEnv
from multiprocessing import Queue
from TrainHandler import TrainHandler
from __env__ import N_TRAJ, NON_PRIVILIGED_DATA, ARS_PATH


class ARSHandler(TrainHandler):
    """
        [TODO]
    """

    def __init__(self, envs: List[GiadogEnv], _continue: bool):
        self.env = envs[0]
        self.model = ARSModel(envs[0], NON_PRIVILIGED_DATA)
        self.results = Queue()

        if _continue: self.model.load_model(ARS_PATH)

    def gen_trajectory(
            self,
            p: Particle, 
            k: int, 
            m: int
        ):
        """
            [TODO]
        """
        t = time()

        # Generate terrain using p
        terrain_file = p.make_terrain(self.env)

        # Train using ARS method
        self.model.update(terrain_file)

        # Compute traverability
        done = False
        cumulative_it_reward = 0
        state = self.env.reset(terrain_file)
        while not done:
            # Get and apply an action.
            action = self.model.get_action(state)
            state, reward, done, _ = self.env.step(action)

            # Get environment reward
            cumulative_it_reward += float(reward)

        if self.env.meta: done = 1
        elif self.env.is_fallen: done = -1
        else: done = 0
        
        travs = self.env.traverability()
        p.traverability[k * N_TRAJ + m] = travs

        # Print trajectory information
        print(
            f'TERRAIN: {p.type} ({p.parameters})| ' +
            f'Done: {done} | ' +
            'Cumulative reward: {:.4f} | '.format(cumulative_it_reward) +
            'Iteration time: {:.4f} | '.format(time() - t) + 
            'Traverability: {:.4f} | '.format(travs) + 
            f'Frames: {len(self.env.trajectory)} \n'
        )

        # Send results
        self.results.put(p)

    def extract_results(self, N: int) -> List[Particle]:
        """
            [TODO]
        """
        # Get data from all trajectories
        return [self.results.get() for _ in range(N * N_TRAJ)]