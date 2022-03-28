"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
from typing import *
from time import time
from uuid import uuid4
from dataclasses import dataclass
from training.GiadogGym import *
from random import randint, choice, choices, uniform
from multiprocessing import Process, Queue, JoinableQueue
from __env__ import N_TRAJ, P_REPLAY, N_EVALUATE, N_PARTICLES, \
    P_TRANSITION, MIN_DESIRED_TRAV, MAX_DESIRED_TRAV, RANDOM_STEP_PROP, \
    ROWS, HILLS_RANGE, STEPS_RANGE, STAIRS_RANGE, LEARNING_RATE, \
    ACTOR_PATH, CRITIC_PATH, GAMMA, BACKTRACK_ITERS, BACKTRACK_COEFF, \
    TRAIN_CRITIC_ITERS, ACTOR_UPDATE_STEPS, CRITIC_UPDATE_STEPS

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
            result_queue: Queue,
            train_method: str
        ):
        Process.__init__(self)
        self.gym_env = gym_env
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.train_method = train_method

    @staticmethod
    def gen_trajectory(
            model,
            gym_env: TeacherEnv,
            p: Particle,
            k: int,
            m: int,
            result_queue: Queue
        ):
        """
            [TODO]
        """
        t = time()
        model.load_models(ACTOR_PATH, CRITIC_PATH)

        buffer_states, buffer_actions, buffer_rewards = [], [], []

        # Generate terrain using C_t
        terrain_file = f'terrains/{p.type}_{uuid4()}.txt'
        gym_env.make_terrain(
            terrain_file,
            p.type,
            rows=ROWS,
            cols=ROWS,
            seed=randint(0, 1e6),
            **p.parameters
        )
        state = gym_env.reset(terrain_file)
        done = False

        cumulative_it_reward = 0
        while not done:
            # Get and apply an action.
            action = model.get_action(state)
            buffer_states.append(state)
            buffer_actions.append(action)
            state, reward, done, _ = gym_env.step(action)

            # Get environment reward
            buffer_rewards.append(reward)
            cumulative_it_reward += float(reward)

        if gym_env.meta: done = 1
        elif gym_env.is_fallen: done = -1
        else: done = 0
        
        travs = gym_env.traverability()
        p.traverability[k * N_TRAJ + m] = travs

        # Print trajectory information
        print(
            f'TERRAIN: {p.type} ({p.parameters})| ' +
            f'Done: {done} | ' +
            'Cumulative reward: {:.4f} | '.format(cumulative_it_reward) +
            'Iteration time: {:.4f} | '.format(time() - t) + 
            'Traverability: {:.4f} | '.format(travs) + 
            f'Frames: {len(gym_env.trajectory)} \n'
        )

        # Send results
        result_queue.put((
            p, 
            buffer_states, 
            buffer_actions, 
            buffer_rewards,
            done
        ))

    def run(self):
        """
            [TODO]
        """
        import tensorflow as tf
        import os

        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        from training.PPO import PPO
        from training.TRPO import TRPO 
        from agents import TeacherNetwork, TeacherValueNetwork
        
        if self.train_method == 'TRPO':
            model = TRPO(
                TeacherNetwork(
                    self.gym_env.action_space, 
                    self.gym_env.observation_space
                ),
                TeacherValueNetwork(self.gym_env.observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )
        elif self.train_method == 'PPO':
            model = PPO(
                TeacherNetwork(
                    self.gym_env.action_space, 
                    self.gym_env.observation_space
                ),
                TeacherValueNetwork(self.gym_env.observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )

        while True:
            # Get task
            p, k, m = self.task_queue.get()
            model.load_models(ACTOR_PATH, CRITIC_PATH)

            self.gen_trajectory(model, self.gym_env, p, k, m, self.result_queue)

            # Send results
            self.task_queue.task_done()

class TerrainCurriculum(object):
    """
        [TODO]
    """
    def __init__(
            self, 
            gym_envs: List[TeacherEnv], 
            train_method: str,
            _continue: bool,
            testing: bool=False
        ):
        """
            [TODO]
        """
        assert len(gym_envs) > 0, 'Must be one or more gym environments.'
        self.gym_envs = gym_envs
        self.train_method = train_method
        self.testing = testing

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

        self.results = Queue()
        if self.testing: 
            pass
        elif len(self.gym_envs) > 1:
            if not _continue:
                p = Process(
                    target=self.__new_model, 
                    args=(
                        gym_envs[0].action_space,
                        gym_envs[0].observation_space,
                        train_method
                    )
                )
                p.start()
                p.join()

            self.tasks = JoinableQueue()
            self.traj_generators = [
                TrajectoryGenerator(
                    env, 
                    self.tasks, 
                    self.results,
                    self.train_method
                ) for env in gym_envs
            ]
            for generator in self.traj_generators: generator.start()
        elif not _continue:
            self.__new_model(
                gym_envs[0].action_space,
                gym_envs[0].observation_space,
                train_method
            )

    @staticmethod
    def __new_model(
            action_space: gym.Space, 
            observation_space: gym.Space,
            train_method: str 
        ):
        """
            [TODO]
        """
        import tensorflow as tf
        import os
        
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

        from training.PPO import PPO
        from training.TRPO import TRPO 
        from agents import TeacherNetwork, TeacherValueNetwork
        
        if train_method == 'TRPO':
            model = TRPO(
                TeacherNetwork(action_space, observation_space),
                TeacherValueNetwork(observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )
        elif train_method == 'PPO':
            model = PPO(
                TeacherNetwork(action_space, observation_space),
                TeacherValueNetwork(observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )

        model.save_models(ACTOR_PATH, CRITIC_PATH)

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
            p.parameters[attr] = np.clip(
                p.parameters[attr],
                RANGES[p.type][attr][0],
                RANGES[p.type][attr][1]
            )
        return p

    @staticmethod
    def __update_model(
            action_space: gym.Space, 
            observation_space: gym.Space,
            train_method: str,
            buffer_states: List[Dict[str, np.array]], 
            buffer_actions: List[np.array], 
            buffer_rewards: List[float], 
            done: bool
        ):
        """
            [TODO]
        """
        import tensorflow as tf
        import os
        
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        
        from training.PPO import PPO
        from training.TRPO import TRPO 
        from agents import TeacherNetwork, TeacherValueNetwork
        
        if train_method == 'TRPO':
            model = TRPO(
                TeacherNetwork(action_space, observation_space),
                TeacherValueNetwork(observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )
        elif train_method == 'PPO':
            model = PPO(
                TeacherNetwork(action_space, observation_space),
                TeacherValueNetwork(observation_space),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
            )
        model.load_models(ACTOR_PATH, CRITIC_PATH)

        # Get critic value
        if done: critic_val = 0
        else: critic_val = model.critic_value(buffer_states[-1])

        # Compute current discount reward
        discounted_reward = []
        for r in buffer_rewards[::-1]:
            critic_val = r + GAMMA * critic_val
            discounted_reward.append(critic_val)
        discounted_reward.reverse()

        # Update actor and critic parameters
        if train_method == 'TRPO':
            model.update(
                buffer_states, 
                buffer_actions, 
                np.array(discounted_reward)[:, np.newaxis], 
                TRAIN_CRITIC_ITERS, 
                BACKTRACK_ITERS, 
                BACKTRACK_COEFF
            )
        elif train_method == 'PPO':
            model.update(
                buffer_states, 
                buffer_actions, 
                np.array(discounted_reward)[:, np.newaxis], 
                ACTOR_UPDATE_STEPS,
                CRITIC_UPDATE_STEPS
            )

        # Save model
        model.save_models(ACTOR_PATH, CRITIC_PATH)

    def train(
            self, 
            artificial_trajectory_gen: Callable=None,
            terrain_to_show: str='hills',
            epochs_to_show: int=50
        ):
        """
            [TODO]
        """
        if self.testing:
            generate = artificial_trajectory_gen
        elif len(self.gym_envs) > 1:
            generate = lambda p, k, m : self.tasks.put((p, k, m))
        else:
            import tensorflow as tf
            import os

            gpu_devices = tf.config.experimental.list_physical_devices('GPU')
            for device in gpu_devices:
                tf.config.experimental.set_memory_growth(device, True)
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

            from training.PPO import PPO
            from training.TRPO import TRPO 
            from agents import TeacherNetwork, TeacherValueNetwork
            
            if self.train_method == 'TRPO':
                model = TRPO(
                    TeacherNetwork(
                        self.gym_envs[0].action_space, 
                        self.gym_envs[0].observation_space
                    ),
                    TeacherValueNetwork(self.gym_envs[0].observation_space),
                    tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
                )
            elif self.train_method == 'PPO':
                model = PPO(
                    TeacherNetwork(
                        self.gym_envs[0].action_space, 
                        self.gym_envs[0].observation_space
                    ),
                    TeacherValueNetwork(self.gym_envs[0].observation_space),
                    tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                    tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
                )

            generate = lambda p, k, m : TrajectoryGenerator.gen_trajectory(
                model,
                self.gym_envs[0],
                p,
                k,
                m,
                self.results
            )

        epoch = 0
        while True:
            for k in range(N_EVALUATE):
                # We add the tasks for the trajectory generators
                for p in self.C_t:
                    for m in range(N_TRAJ):
                        generate(p, k, m)

                # If we are in test mode, no model should be updated
                if self.testing: continue

                # Wait for all of the tasks to finish
                if len(self.gym_envs) > 1: self.tasks.join()
                
                # Get data from all trajectories
                N = len(self.C_t)
                buffers_states, buffers_actions, buffers_rewards = [], [], []
                buffer_done, self.C_t = [], []
                for _ in range(N * N_TRAJ):
                    new_p, states, actions, rewards, done = self.results.get()
                    self.C_t.append(new_p)
                    buffers_states.append(states)
                    buffers_actions.append(actions)
                    buffers_rewards.append(rewards)
                    buffer_done.append(done)

                # Update policy for every iteration
                for i in range(N * N_TRAJ):
                    if len(self.gym_envs) > 1:
                        p = Process(
                            target=self.__update_model,
                            args=(
                                self.gym_envs[0].action_space,
                                self.gym_envs[0].observation_space,
                                self.train_method,
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
                            self.gym_envs[0].action_space,
                            self.gym_envs[0].observation_space,
                            self.train_method,
                            buffers_states[i],
                            buffers_actions[i],
                            buffers_rewards[i],
                            buffer_done[i]
                        )
            
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

            # If we are in test mode, we show the terrain with the best fitness
            if self.testing:
                best_p = None 
                best_fitness = -1 
                for p in [p for p in self.C_t if p.type == terrain_to_show]:
                    if p.weight > best_fitness:
                        best_p = p 
                        best_fitness = p.weight
                
                if self.gym_envs[0].sim.gui and epoch % epochs_to_show == 0:
                    terrain_file = f'terrains/{best_p.type}_{uuid4()}.txt'
                    self.gym_envs[0].make_terrain(
                        terrain_file,
                        best_p.type,
                        rows=ROWS,
                        cols=ROWS,
                        seed=randint(0, 1e6),
                        **best_p.parameters
                    )
                    self.gym_envs[0].reset(terrain_file)

                parameters = {
                    attr: '{:.4f}'.format(best_p.parameters[attr]) \
                    for attr in best_p.parameters
                }
                T = len(best_p.traverability)
                traverability = '{:.4f}'.format(sum(best_p.traverability) / T)
                print(
                    f'\033[1;36m[i]\033[0m Epoch: {epoch} |' +\
                    f'Best parameters: {parameters} | ' +\
                    f'Traverability : {traverability} | '
                    'Weight: {:.4f}'.format(best_p.weight)
                )

            epoch += 1
