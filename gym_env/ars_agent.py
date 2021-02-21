import numpy as np
import random


class ARS_Agent():

    def __init__(self, agent_params):

        """
        agent_params: dict

        agent_params = {
            "step size": float, # a.k.a. learning rate
            "directions sampled by iteration": int,
            "exploration standard deviation noise": float, # Debe ser menor a 1
            "number of top directions to use": int,
            "enviroment": openAI gym enviroment, 
            "train episode steps": int # Numero de steps que cada exploracion tendra H. (N es el nuemro de exploraciones) 
        }


        """
        
        self.α   = agent_params["step size"]
        self.N   = agent_params["directions sampled by iteration"]
        self.ν   = agent_params["exploration standard deviation noise"]
        self.b   = agent_params["number of top directions to use"]
        self.env = agent_params["enviroment"]
        self.train_episode_steps =  agent_params["train episode steps"] 

        self.normalizer = Normalizer(self.env.observation_space.shape[0])


        # Chequeamos que el numero de parametros b sea menor a N
        assert self.b <= self.N
    
        #Inicializamos los parametros de la "red" del agente
        self.θ = np.zeros((self.env.action_space.shape[0], self.env.observation_space.shape[0])) 


    def update_V2(self):

        """
        x: Feature vector of the state
        """

        δ = self.sample_deltas()  

        # Calculamos los pares de recompensas

        r_pos = [self.explore(self.θ + δ_i * self.ν) for δ_i in δ]

        r_neg = [self.explore(self.θ - δ_i * self.ν) for δ_i in δ]

        # Concatenamos las listas de recompensas y calculamos la desviasion
        # Estandar de las recompensas: σ_r
        all_rewards = np.array(r_pos + r_neg)
        σ_r = all_rewards.std()

        # Ordenamos los pares de recompesas junto a su delta δ_i 
        # correspondiente de acuerdo a max(r_p, r_n)

        scores  = {max(r_p, r_n) : (r_p,r_n, δ_i) for (r_p, r_n, δ_i) in zip(r_pos, r_neg, δ)}
        order = sorted(scores.keys(), reverse = True)[:self.b] #Limitamos la cantidad de elementos con el parametro b (bests)
        rollout = [scores[key] for key in order]


        # Actualizamos los pesos

        self.θ += self.α/(self.b * σ_r) * np.sum([ (r_p - r_n) * δ_i for (r_p,r_n, δ_i) in rollout], axis = 0)
    


    def evaluate_policy(self,θ,x):
        """
        x: np.ndarray representing the state feature vector
        θ: np.ndarray representing the policy weight vector
        """ 

        return θ.dot(x)


    def explore(self, weights):
        """
        weights: np.ndarray Representa los pesos que se utilizara la politica para explorar.
        """
        
        state = self.env.reset()
        done = False
        steps = 0
        sum_rewards = 0
        
        while not done and steps < self.train_episode_steps:
            
            self.normalizer.observe(state)
            
            state = self.normalizer.normalize(state)
            
            action = self.evaluate_policy(weights, state)
            
            state, reward, done, _ = self.env.step(action)
            
            reward = max(min(reward, 1), -1)
            sum_rewards += reward
            steps += 1

        return sum_rewards
    

    def sample_deltas(self):

        return [np.random.randn(*self.θ.shape) for _ in range(self.N)]
    


class Normalizer():

    def __init__(self, n_inputs):
        self.n   = np.zeros(n_inputs) # Vector de pasos.
        self.μ   = np.zeros(n_inputs) # Vector de las media
        self.dμ  = np.zeros(n_inputs) # Vector de diferencia de medias
        self.var = np.zeros(n_inputs) # Vector de varianzas

    def observe(self, x):
        """
        Se actualizan la media "μ", la diferencia de la media "dμ" y la varaianza "var"
        """
        self.n += 1. 
        μ_p = self.μ.copy()
        self.μ  += (x - self.μ) / self.n 
        self.dμ += (x - μ_p) * (x - self.μ)
        self.var = (self.dμ / self.n).clip(min=1e-2)

    def normalize(self, x):
        """
        De acuerdo a las observaciones se normaliza el vector de features x.
        """
        obs_μ  = self.μ 
        obs_σ = np.sqrt(self.var)
        return (x - obs_μ) / obs_σ