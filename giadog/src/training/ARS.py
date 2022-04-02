"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    ARS: Augmented Random Search implementation.


    References:
    -----------
        * https://arxiv.org/pdf/1803.07055.pdf
        * https://towardsdatascience.com/introduction-to-augmented-random-search-d8d7b55309bd
"""
import numpy as np
from typing import *
from time import sleep

class ARS():

    def __init__(self, gym_env, state_features,step_size, sample_dir_per_iter, 
                explor_std_dev_noise, top_dir_num, train_ep_steps
                ):

        """ 
        Creates an instance of an ARS agent.
        (Augmented Random Search)

        Arguments
        ---------
        gym_env: gym.Env -> The environment to be used by the agent.

        state_features: List[str] -> The features to be used by the agent.

        step size: float -> Learning rate.

        sample_dir_per_iter: int -> Number of samples per iteration.

        explor_std_dev_noise: float -> The standard deviation of the noise added
                                       to the weights
        
        top_dir_num: int -> Number of top directions to be considered.

        train_ep_steps: int ->  Number of steps the exploration will run for.
        """
        
        self.step_size   = step_size
        self.state_features = state_features
        self.sample_dir_per_iter   = sample_dir_per_iter
        self.explor_std_dev_noise   = explor_std_dev_noise
        self.top_dir_num   = top_dir_num
        self.env = gym_env
        self.train_ep_steps =  train_ep_steps

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


        # Chequeamos que el numero de parametros b sea menor i igual a N
        # We check that the top_dir_num is less than or equal to the sample_dir_per_iter
        assert self.top_dir_num <= self.sample_dir_per_iter
    
        # [ESP] Inicializamos los parametros de la "red" del agente
        # [ENG] Initialize the parameters (weights) of the "net" of the agent
        self.theta = np.zeros((16, self.state_dim)) 


    def update_V2(self, terrain_file):

        """
        Update the policy weights using the ARS algorithm (Update V2)
        
        Arguments
        ---------
        terrain_file: str -> The terrain file to be used for the exploration.
        """

        delta = self.sample_deltas()  

        # [ESP] Calculamos los pares de recompensas
        # [ENG] Calculate the pairs of rewards
        r_pos = [self.explore(self.theta + delta_i * self.explor_std_dev_noise,\
                                terrain_file) for delta_i in delta]

        r_neg = [self.explore(self.theta - delta_i * self.explor_std_dev_noise,\
                            terrain_file) for delta_i in delta]

        # [ESP]
        # Concatenamos las listas de recompensas y calculamos la desviacion
        # Estandar de las recompensas: sigma_r

        # [ENG]
        # Concatenate the lists of rewards and calculate the standard deviation
        # of the rewards: sigma_r
        all_rewards = np.array(r_pos + r_neg)
        sigma_r = all_rewards.std()

        # [ESP]
        # Ordenamos los pares de recompesas junto a su delta delta_i 
        # correspondiente de acuerdo a max(r_p, r_n)

        # [ENG]
        # Sort the pairs of rewards and their corresponding delta delta_i
        # according to max(r_p, r_n)
        
        scores  = {max(r_p, r_n) : (r_p,r_n, delta_i) for (r_p, r_n, delta_i) \
                    in zip(r_pos, r_neg, delta)}
        
        # [ESP]
        # Limitamos la cantidad de elementos con el parametro top_dir_num 
        # [ENG]
        # Limit the number of elements with the parameter top_dir_num 
        order = sorted(scores.keys(), reverse = True)[:self.top_dir_num] 
        rollout = [scores[key] for key in order]

        # [ESP]
        # Actualizamos los pesos
        # [ENG]
        # Update the weights
        self.theta += self.step_size/(self.top_dir_num * sigma_r) * np.sum([ \
            (r_p - r_n) * delta_i for (r_p,r_n, delta_i) in rollout], axis = 0)
    
    def action(self, obs: Dict[str, Any])-> np.array:
        """
        Given an observation, the agent returns an action.

        Arguments
        ---------
        obs: np.ndarray -> The observation from the environment.

        Returns
        -------
        np.array -> The action to be taken.
        """        
        x = self.process_obs(obs)
        x = self.normalizer.normalize(x)
        
        
        return self.theta.dot(x)

    def evaluate_policy(self,theta,obs):
        """
        Evaluate the policy given the weights and the observation.
        (In this case the policy evaluation is just a simple dot product)
        
        Arguments
        ---------
        theta: np.ndarray -> The weights to be used.
        obs: np.ndarray -> The observation to be used. (state feature vector)

        Returns
        -------
        float -> The value of the action given the weights and the observation.
        """ 

        x = self.process_obs(obs)
        x = self.normalizer.normalize(x)
        
        return theta.dot(x)
    
    
    def process_obs(self, obs):
        """
        Process the observation to be used by the agent.

        Arguments
        ---------
        obs: np.ndarray -> The observation from the environment.

        Returns
        -------
        np.ndarray -> The processed observation.
        """
        input_x_t = np.concatenate(
            [np.reshape(obs[feature],-1) for feature in self.state_features]
        )
        
        x = np.nan_to_num(input_x_t)
        
        return x
    
    

    def explore(self, weights, terrain_file):
        """
        Explores the environment with the given weights.

        Arguments
        ---------
        weights: np.ndarray -> The weights to be used in the exploration.

        terrain_file: str -> The terrain file to be used for the exploration.

        Returns
        -------
        float -> The reward obtained in the exploration.
        """
        
        self.env.reset(terrain_file)
        done = False
        steps = 0
        sum_rewards = 0
        state = self.env.get_obs()
        while not done and steps < self.train_ep_steps:
            self.normalizer.observe(self.process_obs(state))         
            action = self.evaluate_policy(weights, state)
            state, reward, done, _ = self.env.step(action)
            
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            steps += 1
            

        return float(sum_rewards)
    

    def sample_deltas(self):
        """
        Sample deltas (to be applied to the weights) from a normal distribution.

        Arguments
        ---------
        None

        Returns
        -------
        deltas: np.ndarray -> The sampled deltas.
        """

        return [np.random.randn(*self.theta.shape) for _ in \
            range(self.sample_dir_per_iter)]
    
    def save_model(self, path):
        """
        Saves the model weights and normalizer.

        Arguments:
        ----------
        path: str -> Path to the file where the parameters will be saved.

        Returns:
        --------
        None
        """
        
        np.savez(
            path,
            theta=self.theta,
            normalizer_n =self.normalizer.n,
            normalizer_mu = self.normalizer.mu,
            normalizer_dmu = self.normalizer.dmu,
            normalizer_var = self.normalizer.var,
        )
    
    def load_model(self, path):
        """
        Loads the model weights and normalizer parameters.

        Arguments:
        ----------
        path: str -> Path to the file where the parameters are saved.

        Returns:
        --------
        None
        """

        data = np.load(path)
        self.theta = data['theta']
        self.normalizer.n = data['normalizer_n']
        self.normalizer.mu = data['normalizer_mu']
        self.normalizer.dmu = data['normalizer_dmu']
        self.normalizer.var = data['normalizer_var']

class Normalizer():

    def __init__(self, n_inputs):
        """
        Normalizer class. Normalizes the observations of the ARS agent

        Arguments:
        ----------
        n_inputs: int -> Number of inputs of the agent
        """
        
        # [ESP] Vector de pasos
        # [ENG] Vector of steps. (i.e Number of steps to normalize)
        self.n   = np.zeros(n_inputs)

        # [ESP] Vector de media
        # [ENG] Vector of mean 
        self.mu   = np.zeros(n_inputs) 

        # [ESP] Vector de diferencia de medias
        # [ENG] Vector of mean differences
        self.dmu  = np.zeros(n_inputs) 

        # [ESP] Vector de varianzas
        # [ENG] Vector of variances
        self.var = np.zeros(n_inputs) 

    def observe(self, x):
        """
        The parameter of the Normalizer class is updated with the new 
        observation.

        Arguments:
        ----------
        x: np.ndarray -> Observation of the agent
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

    def normalize(self, x):
        """
        Acording to the normalizer parameters the observation is normalized.

        Arguments:
        ----------
        x: np.ndarray -> Observation of the agent

        Returns:
        --------
        np.ndarray -> Normalized observation of the agent
        """
        obs_mu  = self.mu 
        obs_sigma = np.sqrt(self.var)
        return (x - obs_mu) / obs_sigma
    
    


