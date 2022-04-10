from uuid import uuid4
from random import randint
from GiadogEnv import GiadogEnv
from dataclasses import dataclass
from typing import List, Dict, Callable
from __env__ import N_PARTICLES, ROWS, COLS
from multiprocessing import Process, Queue, JoinableQueue

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

    def make_terrain(self, gym_env: GiadogEnv) -> str:
        # Generate terrain using p
        terrain_file = f'terrains/aux/{self.type}_{uuid4()}.txt'
        gym_env.make_terrain(
            terrain_file,
            self.type,
            rows=ROWS,
            cols=COLS,
            seed=randint(0, 1e6),
            **self.parameters
        )
        return terrain_file

class TrajectoryGenerator(Process):
    """
        [TODO]
    """
    def __init__(
            self, 
            gym_env: GiadogEnv, 
            task_queue: JoinableQueue,
            result_queue: Queue,
            gen_trajectory: Callable
        ):
        Process.__init__(self)
        self.gym_env = gym_env
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.gen_trajectory = gen_trajectory

    def run(self):
        """
            [TODO]
        """
        while True:
            # Get task
            p, k, m = self.task_queue.get()

            self.gen_trajectory(self.gym_env, p, k, m, self.result_queue)

            # Send results
            self.task_queue.task_done()