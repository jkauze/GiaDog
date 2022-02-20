"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
from typing import *
from src.giadog_gym import *
from src.neural_networks import *
from dataclasses import dataclass
from random import randint, choice, choices, uniform


# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
N_TRAJ           = ENV["TRAIN"]["N_TRAJ"]
P_REPLAY         = ENV["TRAIN"]["P_REPLAY"]
N_EVALUATE       = ENV["TRAIN"]["N_EVALUATE"]
N_PARTICLES      = ENV["TRAIN"]["N_PARTICLES"]
P_TRANSITION     = ENV["TRAIN"]["P_TRANSITION"]
MIN_DESIRED_TRAV = ENV["TRAIN"]["MIN_DESIRED_TRAV"]
MAX_DESIRED_TRAV = ENV["TRAIN"]["MAX_DESIRED_TRAV"]
RANDOM_STEP_PROP = ENV["TRAIN"]["RANDOM_STEP_PROP"]
ROWS             = ENV["SIMULATION"]["ROWS"]
COLS             = ENV["SIMULATION"]["COLS"]
TERRAIN_FILE     = ENV["SIMULATION"]["TERRAIN_FILE"]
HILLS_RANGE      = ENV["HILLS_RANGE"]
STEPS_RANGE      = ENV["STEPS_RANGE"]
STAIRS_RANGE     = ENV["STAIRS_RANGE"]

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

@dataclass
class particle:
    type: str
    parameters: Dict[str, float]
    traverability: List[float]
    weight: float=1 / N_PARTICLES
    measurement_prob: float=0.0

    def copy(self):
        return particle(
            self.type,
            self.parameters.copy(),
            self.traverability.copy(),
            self.weight,
            self.measurement_prob
        )

    def __str__(self):
        string = f'(type: {self.type}, ' + '{'
        for attr in self.parameters: string += f'{attr}: {self.parameters[attr]}, '
        string += '}, m_prob: ' + str(self.weight) + ')'

        return string

class terrain_curriculum:
    """
        [TODO]
    """
    def __init__(self, gym_env: teacher_giadog_env, model: teacher_nn):
        self.gym_env = gym_env 
        self.model = model

        hills_C_t  = [
            particle('hills', INIT_VALUES['hills'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        steps_C_t  = [
            particle('steps', INIT_VALUES['steps'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        stairs_C_t = [
            particle('stairs', INIT_VALUES['stairs'].copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        self.C_t = hills_C_t + steps_C_t + stairs_C_t

        self.C_t_history  = [[p.copy() for p in self.C_t]]

    def __compute_measurement_probs(self):
        """
            [TODO]
        """
        for p in self.C_t: 
            p.measurement_prob = sum(
                int(MIN_DESIRED_TRAV <= t <= MAX_DESIRED_TRAV)
                for t in p.traverability
            ) / (N_TRAJ * N_EVALUATE)

    def __compute_traverability(self):
        """
            [TODO]
        """
        done = False
        obs = self.gym_env.get_obs()
        while not done:
            # Obtenemos la accion de la politica
            action = self.model.predict_obs(obs)
            # Aplicamos la accion al entorno
            obs, reward, done, info = self.gym_env.step(action)

        return self.gym_env.traverability()

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

    def __random_walk(self, p: particle) -> particle:
        """
            [TODO]
        """
        for attr in p.parameters:
            step = (RANGES[p.type][attr][1] - RANGES[p.type][attr][0]) * RANDOM_STEP_PROP

            p.parameters[attr] += uniform(-step, step)
            p.parameters[attr] = max(RANGES[p.type][attr][0], p.parameters[attr])
            p.parameters[attr] = min(RANGES[p.type][attr][1], p.parameters[attr])
        return p

    def train(self):
        """
            [TODO]
        """
        while True:
            for k in range(N_EVALUATE):
                for p in self.C_t:
                    for m in range(N_TRAJ):
                        # Generate terrain using C_t
                        self.gym_env.make_terrain(
                            p.type,
                            rows=ROWS,
                            cols=ROWS,
                            seed=randint(0, 1e6),
                            **p.parameters
                        )
                        self.gym_env.reset(TERRAIN_FILE)

                        # Run policy and compute traverability
                        index = k * N_EVALUATE + m
                        p.traverability[index] = self.__compute_traverability()
                        print(f'Traverability: {p.traverability[index]}')
                # Update policy using TRPO
                # [TODO]
            
            # Compute measurement probabilities
            self.__compute_measurement_probs()

            # Update weights w_j
            self.__update_weights()

            # Resample N_particle parameters
            self.__resample()

            # Append parameters to the replay memory
            self.C_t_history.append([p.copy() for p in self.C_t])

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
