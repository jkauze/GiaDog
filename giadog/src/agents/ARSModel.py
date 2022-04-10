"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    ARS: Augmented Random Search implementation.

    References:
    -----------
        * https://arxiv.org/pdf/1803.07055.pdf
        * https://towardsdatascience.com/introduction-to-augmented-random-search-d8d7b55309bd
"""
import gym
import numpy as np
from typing import List, Dict, Any


class Normalizer(object):

    def __init__(self, n_inputs: int):
        """
            Normalizer class. Normalizes the observations of the ARS agent

            Arguments:
            ----------
                n_inputs: int
                    Number of inputs of the agent
        """
        # Vector of steps. (i.e Number of steps to normalize)
        self.n   = np.zeros(n_inputs)
        # Vector of mean 
        self.mu  = np.zeros(n_inputs) 
        # Vector of mean differences
        self.dmu = np.zeros(n_inputs) 
        # Vector of variances
        self.var = np.zeros(n_inputs) 

    def observe(self, x: np.array):
        """
            The parameter of the Normalizer class is updated with the new 
            observation.

            Arguments:
            ----------
                x: numpy.array
                    Observation of the agent
        """
        # Increment the number of observations
        self.n += 1. 
        # Calculate the previous mean
        mu_p = self.mu.copy() 
        # Actualize the mean
        self.mu  += (x - self.mu) / self.n 
        # Update the mean difference 
        self.dmu += (x - mu_p) * (x - self.mu)

        # Update the variance (clip for numerical stability)
        self.var = (self.dmu / self.n).clip(min=1e-2)

    def normalize(self, x: np.array) -> np.array:
        """
            Acording to the normalizer parameters the observation is normalized.

            Arguments:
            ----------
                x: np.array
                    Observation of the agent

            Returns:
            --------
                np.array
                    Normalized observation of the agent
        """
        obs_mu  = self.mu 
        obs_sigma = np.sqrt(self.var)
        return (x - obs_mu) / obs_sigma

class ARSModel(object):
    """
        [TODO]
    """
    def __init__(
            self, 
            gym_env: gym.Env, 
            state_features: List[str],
            step_size: float=0.02, 
            sample_dir_per_iter: int=1, 
            explor_std_dev_noise: float=0.03, 
            top_dir_num: int=1, 
            train_ep_steps: int=2000
        ):
        """ 
            Creates an instance of an ARS (Augmented Random Search) agent.

            Arguments
            ---------
                gym_env: gym.Env
                    The environment to be used by the agent.

                state_features: List[str]
                    The features to be used by the agent.

                step size: float, optional
                    Learning rate.
                    Default: 0.02

                sample_dir_per_iter: int, optional
                    Number of samples per iteration.
                    Default: 1

                explor_std_dev_noise: float, optional
                    The standard deviation of the noise added to the weights
                    Default: 0.03
                
                top_dir_num: int, optional
                    Number of top directions to be considered.
                    Default: 1

                train_ep_steps: int, optional
                    Number of steps the exploration will run for.
                    Default: 2000
        """
        # We check that the top_dir_num is less than or equal to the 
        # sample_dir_per_iter
        assert top_dir_num <= sample_dir_per_iter

        self.step_size = step_size
        self.state_features = state_features
        self.sample_dir_per_iter = sample_dir_per_iter
        self.explor_std_dev_noise = explor_std_dev_noise
        self.top_dir_num = top_dir_num
        self.env = gym_env
        self.train_ep_steps = train_ep_steps

        # from the enviroment we get the dimensions of the state using the 
        # state_features list
        self.state_dim  = 0
        for key in self.state_features:
            elem = self.env.observation_space[key].shape
            if len(elem) == 0:
                self.state_dim  += 1
            else:
                dims = [elem[i] for i in range(len(elem))]
                self.state_dim  += np.prod(dims)
        
        self.normalizer = Normalizer(self.state_dim)

        # Initialize the parameters (weights) of the "net" of the agent
        self.theta = np.zeros((16, self.state_dim)) 

    def __process_obs(self, obs: np.array) -> np.array:
        """
            Process the observation to be used by the agent.

            Arguments
            ---------
                obs: numpy.array
                    The observation from the environment.

            Returns
            -------
                numpy.array
                    The processed observation.
        """
        input_x_t = np.concatenate(
            [np.reshape(obs[feature],-1) for feature in self.state_features]
        )
        return np.nan_to_num(input_x_t)

    def __evaluate_policy(self, theta: np.array, obs: np.array) -> float:
        """
            Evaluate the policy given the weights and the observation.
            (In this case the policy evaluation is just a simple dot product)
            
            Arguments
            ---------
                theta: numpy.array
                    The weights to be used.

                obs: numpy.array
                    The observation to be used. (state feature vector)

            Returns
            -------
                float
                    The value of the action given the weights and the 
                    observation.
        """ 
        return theta.dot(self.normalizer.normalize(self.__process_obs(obs)))

    def __sample_deltas(self) -> np.array:
        """
            Sample deltas (to be applied to the weights) from a normal 
            distribution.

            Returns
            -------
                numpy.array -> The sampled deltas.
        """

        return [
            np.random.randn(*self.theta.shape) 
            for _ in range(self.sample_dir_per_iter)
        ]

    def __explore(self, weights: np.array, terrain_file: str):
        """
            Explores the environment with the given weights.

            Arguments
            ---------
                weights: numpy.array
                    The weights to be used in the exploration.

                terrain_file: str
                    The terrain file to be used for the exploration.

            Returns
            -------
                float
                    The reward obtained in the exploration.
        """
        self.env.reset(terrain_file)
        done = False
        steps = 0
        sum_rewards = 0
        state = self.env.get_obs()

        while not done and steps < self.train_ep_steps:
            self.normalizer.observe(self.__process_obs(state))         
            action = self.__evaluate_policy(weights, state)
            state, reward, done, _ = self.env.step(action)
            
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            steps += 1

        return float(sum_rewards)

    def save_model(self, path: str):
        """
            Saves the model weights and normalizer.

            Arguments:
            ----------
                path: str
                    Path to the file where the parameters will be saved.
        """
        np.savez(
            path,
            theta=self.theta,
            normalizer_n =self.normalizer.n,
            normalizer_mu = self.normalizer.mu,
            normalizer_dmu = self.normalizer.dmu,
            normalizer_var = self.normalizer.var,
        )
    
    def load_model(self, path: str):
        """
            Loads the model weights and normalizer parameters.

            Arguments:
            ----------
                path: str
                    Path to the file where the parameters are saved.
        """
        data = np.load(path)
        self.theta = data['theta']
        self.normalizer.n = data['normalizer_n']
        self.normalizer.mu = data['normalizer_mu']
        self.normalizer.dmu = data['normalizer_dmu']
        self.normalizer.var = data['normalizer_var']

    def get_action(self, obs: Dict[str, Any]) -> np.array:
        """
            Given an observation, the agent returns an action.

            Arguments
            ---------
                obs: numpy.array
                    The observation from the environment.

            Returns
            -------
                numpy.array
                    The action to be taken.
        """        
        x = self.__process_obs(obs)
        x = self.normalizer.normalize(x)
        
        
        return self.theta.dot(x)

    def update(self, terrain_file: str):
        """
            Update the policy weights using the ARS algorithm (Update V2)
            
            Arguments
            ---------
                terrain_file: str
                    Path to the .txt file representing the terrain.
        """
        delta = self.__sample_deltas()  

        # Calculate the pairs of rewards
        explore = lambda delta: self.__explore(
            self.theta + delta * self.explor_std_dev_noise,
            terrain_file
        )
        r_pos = [explore(delta_i) for delta_i in delta]
        r_neg = [explore(-delta_i) for delta_i in delta]

        # Concatenate the lists of rewards and calculate the standard deviation
        # of the rewards: sigma_r
        all_rewards = np.array(r_pos + r_neg)
        sigma_r = all_rewards.std()

        # Sort the pairs of rewards and their corresponding delta delta_i
        # according to max(r_p, r_n)
        scores  = {
            max(r_p, r_n) : (r_p,r_n, delta_i) 
            for (r_p, r_n, delta_i) in zip(r_pos, r_neg, delta)
        }
        
        # Limit the number of elements with the parameter top_dir_num 
        order = sorted(scores.keys(), reverse = True)[:self.top_dir_num] 
        rollout = [scores[key] for key in order]

        # Update the weights
        self.theta += self.step_size / (self.top_dir_num * sigma_r) * \
            np.sum(
                [(r_p - r_n) * delta_i for (r_p,r_n, delta_i) in rollout], 
                axis=0
            )
    
    


