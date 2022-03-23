import numpy as np
from typing import *


class ars_agent():

    def __init__(self, agent_params):

        """
        Creates an instance of an ARS agent.
        (Augmented Random Search)

        Arguments
        ---------

        agent_params: dict. Dictionary of the agent parameters.

        The parameters are the following:

        agent_params = {
            "step size"                      : float, 
                * a.k.a. learning rate
            
            "directions sampled by iteration": int,
                * number of directions sampled by iteration
            
            "exploration standard deviation noise": float,
                * The standard deviation of the noise added to the weights
                  must be < 1.0 and > 0.0
            
            "number of top directions to use": int,
                * number of top directions to use for updating the policy
            
            "enviroment": openAI gym enviroment, 
                * Gym enviroment to be used to train the agent
            
            "train episode steps": int 
                * Number of steps the exploration will run for.


        """
        
        self.alpha   = agent_params["step size"]
        self.N   = agent_params["directions sampled by iteration"]
        self.nu   = agent_params["exploration standard deviation noise"]
        self.b   = agent_params["number of top directions to use"]
        self.env = agent_params["enviroment"]
        self.train_episode_steps =  agent_params["train episode steps"] 

        self.priviliged_data = self.env.PRIVILIGED_DATA

        self.non_priviliged_data = self.env.NON_PRIVILIGED_DATA

        
        self.normalizer = Normalizer(145+59)


        # Chequeamos que el numero de parametros b sea menor a N
        assert self.b <= self.N
    
        #Inicializamos los parametros de la "red" del agente
        self.theta = np.zeros((16, 145+59)) 


    def update_V2(self, terrain_file):

        """
        x: Feature vector of the state
        """

        delta = self.sample_deltas()  

        # [ESP] Calculamos los pares de recompensas
        # [ENG] Calculate the pairs of rewards
        r_pos = [self.explore(self.theta + delta_i * self.nu, terrain_file) for 
                                delta_i in delta]

        r_neg = [self.explore(self.theta - delta_i * self.nu, terrain_file) for 
                                delta_i in delta]

        # Concatenamos las listas de recompensas y calculamos la desviasion
        # Estandar de las recompensas: sigma_r
        all_rewards = np.array(r_pos + r_neg)
        sigma_r = all_rewards.std()

        # Ordenamos los pares de recompesas junto a su delta delta_i 
        # correspondiente de acuerdo a max(r_p, r_n)

        scores  = {max(r_p, r_n) : (r_p,r_n, delta_i) for (r_p, r_n, delta_i) in zip(r_pos, r_neg, delta)}
        order = sorted(scores.keys(), reverse = True)[:self.b] #Limitamos la cantidad de elementos con el parametro b (bests)
        rollout = [scores[key] for key in order]


        # Actualizamos los pesos

        self.theta += self.alpha/(self.b * sigma_r) * np.sum([ (r_p - r_n) * delta_i for (r_p,r_n, delta_i) in rollout], axis = 0)
    
    def action(self, obs: Dict[str, Any])-> np.array:
        """
        Given an observation, the agent returns an action.
        """        
        x = self.process_obs(obs)
        x = self.normalizer.normalize(x)
        
        return self.theta.dot(x)

    def evaluate_policy(self,theta,obs):
        """
        x: np.ndarray representing the state feature vector
        theta: np.ndarray representing the policy weight vector
        """ 

        x = self.process_obs(obs)
        x = self.normalizer.normalize(x)
        
        return theta.dot(x)
    
    
    def process_obs(self, obs):
        """
        obs: dict
        """
        input_x_t = np.concatenate(
            [np.reshape(obs[data],-1) for data in self.priviliged_data]
        )
        
        input_o_t = np.concatenate(
            [np.reshape(obs[data],-1) for data in self.non_priviliged_data]
        )

        x = np.concatenate((input_x_t, input_o_t)).flatten()
        
        x = np.nan_to_num(x)
        
        return x
    
    

    def explore(self, weights, terrain_file):
        """
        weights: np.ndarray Representa los pesos que se utilizara la politica para explorar.
        """
        
        self.env.reset(terrain_file)
        done = False
        steps = 0
        sum_rewards = 0
        state = self.env.get_obs()
        while not done and steps < self.train_episode_steps:
            self.normalizer.observe(self.process_obs(state))         
            action = self.evaluate_policy(weights, state)
            state, reward, done, _ = self.env.step([action])
            
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            steps += 1
        
        return sum_rewards
    

    def sample_deltas(self):

        return [np.random.randn(*self.theta.shape) for _ in range(self.N)]

class Normalizer():

    def __init__(self, n_inputs):
        self.n   = np.zeros(n_inputs) # Vector de pasos.
        self.mu   = np.zeros(n_inputs) # Vector de las media
        self.dmu  = np.zeros(n_inputs) # Vector de diferencia de medias
        self.var = np.zeros(n_inputs) # Vector de varianzas

    def observe(self, x):
        """
        Se actualizan la media "mu", la diferencia de la media "dmu" y la varaianza "var"
        """
        self.n += 1. 
        mu_p = self.mu.copy()
        self.mu  += (x - self.mu) / self.n 
        self.dmu += (x - mu_p) * (x - self.mu)
        self.var = (self.dmu / self.n).clip(min=1e-2)

    def normalize(self, x):
        """
        De acuerdo a las observaciones se normaliza el vector de features x.
        """
        obs_mu  = self.mu 
        obs_sigma = np.sqrt(self.var)
        return (x - obs_mu) / obs_sigma
