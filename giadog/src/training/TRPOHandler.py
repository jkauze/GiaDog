"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Proximal Policy Optimization (with support for Natural Policy Gradient).

    This imlementation is a modification of the RLZoo implementation of PPO

    References:
    -----------
        * [TODO]
"""
# Utilities
import numpy as np
from time import time
from typing import List, Dict
from utils import Particle, TrajectoryGenerator
from multiprocessing import Process, Queue, JoinableQueue

# Train
import gym
from GiadogEnv import GiadogEnv
from TrainHandler import TrainHandler
from __env__ import ACTOR_PATH, CRITIC_PATH, N_TRAJ, GAMMA, \
    ACTOR_UPDATE_STEPS, CRITIC_UPDATE_STEPS


class TRPOHandler(TrainHandler):
    """
        [TODO]
    """

    def __init__(self, envs: List[GiadogEnv], _continue: bool):
        self.envs = envs
        self._continue = _continue

        self.results = Queue()
        self.tasks = JoinableQueue()
        self.parallel = len(self.envs) > 1

        if self.parallel:
            self.traj_generators = [
                TrajectoryGenerator(
                    env, 
                    self.tasks, 
                    self.results,
                    self.__trajectory
                ) for env in self.envs
            ]
            for generator in self.traj_generators: generator.start()
        
        if not _continue and self.parallel:
            p = Process(
                target=self.__new_model, 
                args=(self.envs[0].action_space, self.envs[0].observation_space)
            )
            p.start()
            p.join()
        elif not _continue:
            self.__new_model(self.envs[0].action_space, self.envs[0].observation_space)

    @staticmethod
    def __new_model(action_space: gym.Space, observation_space: gym.Space):
        """
            [TODO]
        """
        from agents import TRPOModel
        
        # Initialize new models and optimizers
        model = TRPOModel(action_space, observation_space)

        model.save_models(ACTOR_PATH, CRITIC_PATH)

    @staticmethod
    def __update_model(
            action_space: gym.Space, 
            observation_space: gym.Space,
            buffer_states: List[Dict[str, np.array]], 
            buffer_actions: List[np.array], 
            buffer_rewards: List[float], 
            done: int
        ):
        """
            [TODO]
        """
        from agents import TRPOModel
        
        model = TRPOModel(action_space, observation_space)
        model.load_models(ACTOR_PATH, CRITIC_PATH)

        # Get critic value
        if done == 1: critic_val = 0
        else: critic_val = model.critic_value(buffer_states[-1])

        # Compute current discount reward
        discounted_reward = []
        for r in buffer_rewards[::-1]:
            critic_val = r + GAMMA * critic_val
            discounted_reward.append(critic_val)
        discounted_reward.reverse()

        # Update actor and critic parameters
        model.update(
            buffer_states, 
            buffer_actions, 
            np.array(discounted_reward)[:, np.newaxis], 
            ACTOR_UPDATE_STEPS,
            CRITIC_UPDATE_STEPS
        )

        # Save model
        model.save_models(ACTOR_PATH, CRITIC_PATH)

    @staticmethod
    def __trajectory(
            env: GiadogEnv,
            p: Particle, 
            k: int, 
            m: int, 
            results: Queue,
        ):
        """
            [TODO]
        """
        t = time()

        from agents import TRPOModel
        model = TRPOModel(env.action_space, env.observation_space)
        model.load_models(ACTOR_PATH, CRITIC_PATH)

        buffer_states, buffer_actions, buffer_rewards = [], [], []

        # Generate terrain using p
        terrain_file = p.make_terrain(env)
        state = env.reset(terrain_file)

        done = False
        cumulative_it_reward = 0
        while not done:
            # Get and apply an action.
            action = model.get_action(state)
            buffer_states.append(state)
            buffer_actions.append(action)
            state, reward, done, _ = env.step(action)

            # Get environment reward
            buffer_rewards.append(reward)
            cumulative_it_reward += float(reward)

        if env.meta: done = 1
        elif env.is_fallen: done = -1
        else: done = 0
        
        travs = env.traverability()
        p.traverability[k * N_TRAJ + m] = travs

        # Print trajectory information
        print(
            f'TERRAIN: {p.type} ({p.parameters})| ' +
            f'Done: {done} | ' +
            'Cumulative reward: {:.4f} | '.format(cumulative_it_reward) +
            'Iteration time: {:.4f} | '.format(time() - t) + 
            'Traverability: {:.4f} | '.format(travs) + 
            f'Frames: {len(env.trajectory)} \n'
        )

        # Send results
        results.put((
            p, 
            buffer_states, 
            buffer_actions, 
            buffer_rewards,
            done
        ))

    def gen_trajectory(
            self,
            p: Particle, 
            k: int, 
            m: int
        ):
        """
            [TODO]
        """
        if self.parallel: self.tasks.put((p, k, m))
        else: self.__trajectory(self.envs[0], p, k, m, self.results)

    def extract_results(self, N: int) -> List[Particle]:
        """
            [TODO]
        """
        # Wait for all of the tasks to finish
        if self.parallel: self.tasks.join()
        
        # Get data from all trajectories
        buffers_states, buffers_actions, buffers_rewards = [], [], []
        buffer_done, C_t = [], []
        for _ in range(N * N_TRAJ):
            new_p, states, actions, rewards, done = self.results.get()
            C_t.append(new_p)
            buffers_states.append(states)
            buffers_actions.append(actions)
            buffers_rewards.append(rewards)
            buffer_done.append(done)

        # Update policy for every iteration
        for i in range(N * N_TRAJ):
            if self.parallel:
                p = Process(
                    target=self.__update_model,
                    args=(
                        self.envs[0].action_space,
                        self.envs[0].observation_space,
                        buffers_states[i],
                        buffers_actions[i],
                        buffers_rewards[i],
                        buffer_done[i]
                    )
                )
                p.start()
                p.join()
            else:
                self.__update_model(
                    self.envs[0].action_space,
                    self.envs[0].observation_space,
                    buffers_states[i],
                    buffers_actions[i],
                    buffers_rewards[i],
                    buffer_done[i]
                )
    
        return C_t
