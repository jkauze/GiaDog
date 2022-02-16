"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
from typing import *
from random import randint, choice, choices, uniform
from src.giadog_gym import *
from dataclasses import dataclass


# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
N_TRAJ            = ENV["TRAIN"]["N_TRAJ"]
P_REPLAY          = ENV["TRAIN"]["P_REPLAY"]
N_EVALUATE        = ENV["TRAIN"]["N_EVALUATE"]
N_PARTICLES       = ENV["TRAIN"]["N_PARTICLES"]
P_TRANSITION      = ENV["TRAIN"]["P_TRANSITION"]
MIN_DESIRED_TRAV  = ENV["TRAIN"]["MIN_DESIRED_TRAV"]
MAX_DESIRED_TRAV  = ENV["TRAIN"]["MAX_DESIRED_TRAV"]
RANDOM_STEP_RANGE = ENV["TRAIN"]["RANDOM_STEP_RANGE"]
ROWS              = ENV["SIMULATION"]["ROWS"]
COLS              = ENV["SIMULATION"]["COLS"]
TERRAIN_FILE      = ENV["SIMULATION"]["TERRAIN_FILE"]
HILLS_INIT        = ENV["HILLS_INIT"]
STEPS_INIT        = ENV["STEPS_INIT"]
STAIRS_INIT       = ENV["STAIRS_INIT"]

@dataclass
class particle:
    parameters: Dict[str, float]
    traverability: List[float]
    weight: float=1 / N_PARTICLES
    measurement_prob: float=0.0

    def copy(self):
        return particle(
            self.parameters,
            self.traverability,
            self.weight,
            self.measurement_prob
        )

class terrain_curriculum:
    """
        [TODO]
    """
    def __init__(self, gym_env: teacher_giadog_env):
        self.gym_env = gym_env 
        self.hills_C_t  = [
            particle(HILLS_INIT.copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        self.steps_C_t  = [
            particle(STEPS_INIT.copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        self.stairs_C_t = [
            particle(STAIRS_INIT.copy(), [0] * N_TRAJ * N_EVALUATE) 
            for _ in range(N_PARTICLES)
        ]
        self.particles = [
            (self.hills_C_t, 'hills'),
            (self.steps_C_t, 'steps'),
            (self.stairs_C_t, 'stairs')
        ]
        self.hills_C_t_history  = [self.hills_C_t]
        self.steps_C_t_history  = [self.steps_C_t]
        self.stairs_C_t_history = [self.stairs_C_t]

    def __compute_measurement_probs(self):
        """
            [TODO]
        """
        for p in self.hills_C_t + self.steps_C_t + self.stairs_C_t: 
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
            action = self.gym_env.predict(obs)
            # Aplicamos la accion al entorno
            obs, reward, done, info = self.gym_env.step(action)

        return self.gym_env.traverability()

    def __resample(self):
        """
            [TODO]
        """
        self.hills_C_t = choices(
            self.hills_C_t, 
            [p.measurement_prob for p in self.hills_C_t], 
            k=N_PARTICLES
        )
        self.steps_C_t = choices(
            self.steps_C_t, 
            [p.measurement_prob for p in self.steps_C_t], 
            k=N_PARTICLES
        )
        self.stairs_C_t = choices(
            self.stairs_C_t, 
            [p.measurement_prob for p in self.stairs_C_t], 
            k=N_PARTICLES
        )
        self.particles = [
            (self.hills_C_t, 'hills'),
            (self.steps_C_t, 'steps'),
            (self.stairs_C_t, 'stairs')
        ]

    def __random_walk(self, p: particle) -> particle:
        """
            [TODO]
        """
        for attr in p.parameters:
            p.parameters[attr] += uniform(-RANDOM_STEP_RANGE, RANDOM_STEP_RANGE)
        return p

    def train(self):
        """
            [TODO]
        """
        while True:
            for k in N_EVALUATE:
                for C_t, terrain_type in self.particles:
                    for l in N_PARTICLES:
                        for m in N_TRAJ:
                            # Generate terrain using C_t
                            self.gym_env.make_terrain(
                                terrain_type,
                                rows=ROWS,
                                cols=ROWS,
                                seed=randint(0, 1e6)
                                **C_t[l].parameters
                            )
                            self.gym_env.reset(TERRAIN_FILE)

                            # Run policy and compute traverability
                            index = k * N_EVALUATE + m
                            C_t[l].traverability[index] = self.__compute_traverability()
                # Update policy using TRPO
                # [TODO]
            
            # Compute measurement probabilities
            self.__compute_measurement_probs()

            # Update weights w_j
            for C_t, _ in self.particles:
                prob_sum = sum(p.measurement_prob for p in C_t)
                for p in C_t: p.weight = p.measurement_prob / prob_sum

            # Resample N_particle parameters
            self.__resample()

            # Append parameters to the replay memory
            self.hills_C_t_history.append([p.copy() for p in self.hills_C_t])
            self.steps_C_t_history.append([p.copy() for p in self.steps_C_t])
            self.stairs_C_t_history.append([p.copy() for p in self.stairs_C_t])

            for i in range(len(self.hills_C_t)):
                prob = randint(0, 100) / 100

                # By p_replay probability, sample from replay memory
                if prob <= P_REPLAY:
                    self.hills_C_t[i] = choice(choice(self.hills_C_t_history))
                # By ptransition probability, move C_T to an adjacent value in C.
                elif P_REPLAY < prob <= P_REPLAY + P_TRANSITION:
                    self.hills_C_t[i] = self.__random_walk(self.hills_C_t[i])
                
