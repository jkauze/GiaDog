"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
from typing import *
from dataclasses import dataclass
from src.training.GiadogGym import *
from random import randint, choice, choices, uniform
from multiprocessing import Process, Queue, JoinableQueue
from src.__env__ import N_TRAJ, P_REPLAY, N_EVALUATE, N_PARTICLES, \
    P_TRANSITION, MIN_DESIRED_TRAV, MAX_DESIRED_TRAV, RANDOM_STEP_PROP, \
    ROWS, TERRAIN_FILE, HILLS_RANGE, STEPS_RANGE, STAIRS_RANGE

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
class Particle(object):
    type: str
    parameters: Dict[str, float]
    traverability: List[float]
    weight: float=1 / N_PARTICLES
    measurement_prob: float=0.0

    def copy(self):
        return Particle(
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

class TrajectoryGenerator(Process):
    """
        [TODO]
    """
    def __init__(
            self, 
            gym_env: TeacherEnv, 
            task_queue: JoinableQueue,
            result_queue: Queue
        ):
        Process.__init__(self)
        self.gym_env = gym_env
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        from src.agents import TeacherNetwork
        
        model = TeacherNetwork(
            self.gym_env.action_space, 
            self.gym_env.observation_space
        )

        while True:
            # Get task
            p, k, m = self.task_queue.get()
            model.load('model_checkpoint/')

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
            done = False
            obs = self.gym_env.get_obs()
            while not done:
                # Obtenemos la accion de la politica
                action = model.predict_obs(obs)
                # Aplicamos la accion al entorno
                obs, reward, done, info = self.gym_env.step(action)

            p.traverability[k * N_EVALUATE + m] = self.gym_env.traverability()
            trajectory = self.gym_env.trajectory
            print(f'Traverability: {p.traverability[k * N_EVALUATE + m]}')

            self.task_queue.task_done()
            self.result_queue.put((p, trajectory))

class TerrainCurriculum(object):
    """
        [TODO]
    """
    def __init__(self, gym_envs: List[TeacherEnv]):
        assert len(gym_envs) > 0, 'Must be one or more gym environments.'

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

        p = Process(
            target=self.__new_model, 
            args=(gym_envs[0].action_space, gym_envs[0].observation_space)
        )
        p.start()
        p.join()

        self.tasks = JoinableQueue()
        self.results = Queue()
        self.traj_generators = [
            TrajectoryGenerator(env, self.tasks, self.results) for env in gym_envs
        ]
        for generator in self.traj_generators: generator.start()

    @staticmethod
    def __new_model(action_space, observation_space):
        """
            [TODO]
        """
        from src.agents import TeacherNetwork
        TeacherNetwork(
            action_space, 
            observation_space
        ).save('model_checkpoint/')

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
            p.parameters[attr] = max(RANGES[p.type][attr][0], p.parameters[attr])
            p.parameters[attr] = min(RANGES[p.type][attr][1], p.parameters[attr])
        return p

    def train(self):
        """
            [TODO]
        """
        while True:
            for k in range(N_EVALUATE):
                # We add the tasks for the trajectory generators
                for p in self.C_t:
                    for m in range(N_TRAJ):
                        self.tasks.put((p, k, m))

                # Wait for all of the tasks to finish
                self.tasks.join()
                
                N = len(self.C_t)
                self.C_t = []
                self.trajectories = []
                for _ in range(N * N_TRAJ):
                    new_p, trajectory = self.results.get()
                    self.C_t.append(new_p)
                    self.trajectories.append(trajectory)

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
