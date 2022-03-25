"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Proximal Policy Optimization (with support for Natural Policy Gradient).

    This imlementation is a modification of the RLZoo implementation of PPO

    References:
    -----------
        * [TODO]
"""
import time
import numpy as np
import tensorflow as tf
from agents import *
from __env__ import EPSILON
from training.GiadogGym import TeacherEnv


class PPO(object):
    """
        This class implements the PPO clip algorithm.
    
        References:
        -----------
            * https://arxiv.org/abs/1707.06347
    """

    def __init__(
            self, 
            actor: TeacherNetwork,
            critic: TeacherValueNetwork,
            actor_optimizer: tf.keras.optimizers.Optimizer, 
            critic_optimizer: tf.keras.optimizers.Optimizer, 
            epsilon: float=0.2
        ):
        """
            Arguments
            ---------
                actor: TeacherNetwork
                    Neural network that works as policy.

                critic: TeacherValueNetwork
                    Neural network that works as critic.

                actor_optimizer: tensorflow.keras.optimizers.Optimizer
                    Critic Training Optimizer.
                
                critic_optimizer: tensorflow.keras.optimizers.Optimizer
                    Critic Training Optimizer.

                epsilon: float, optional
                    CLIP parameter.
                    Default: 0.2
        """
        self.name = 'PPO'
        self.actor = actor 
        self.critic = critic
        self.critic_opt = critic_optimizer 
        self.actor_opt = actor_optimizer
        self.epsilon = epsilon

    def save_models(self, actor_path: str, critic_path: str):
        """
            Save actor and critic models.

            Arguments:
            ----------
                actor_path: str
                    Path where the actor model will be saved.

                critic_path: str
                    Path where the critic model will be saved.
        """
        self.actor.save(actor_path)
        self.critic.save(critic_path)

    def load_models(self, actor_path: str, critic_path: str):
        """
            Load actor and critic models.

            Arguments:
            ----------
                actor_path: str
                    Path where the actor model was saved.

                critic_path: str
                    Path where the critic model was saved.
        """
        self.actor.load(actor_path)
        self.critic.load(critic_path)

    def actor_train(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            old_actor_prob: tf.Tensor
        ):
        """
            Update policy (actor) network.
            
            Arguments:
            ---------
                states: List[Dict[str, np.array]]
                    State list.

                actions: List[np.array]
                    List of actions corresponding to the states.

                advantage: np.array
                    List of advantage obtained in each state.

                old_actor_prob: tensorflow.Tensor
                    Probability obtained by the previous policy.
        """
        with tf.GradientTape() as tape:
            self.actor(states)

            # We compute the probability of the actions taken by the actor
            actor_prob = tf.exp(self.actor.policy_dist.logp(actions))
            # Calculate the ratio between the old and the new policy
            ratio = actor_prob / (old_actor_prob + EPSILON)

            # Calculate the surrogate loss. pi_new/pi_old * advantage
            surr = ratio * advantage
            
            # Apply the clipped surrogate function (check the PPO clip update) 
            actor_loss = -tf.reduce_mean(tf.minimum(
                surr, 
                tf.clip_by_value(
                    ratio, 
                    1. - self.epsilon, 
                    1. + self.epsilon
                ) * advantage
            ))

        # Compute the gradient of the loss with respect to the policy network
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_weights
        )
        # Apply the gradient to the policy network
        # (See that the apply gradient is used with the optimizer)
        self.actor_opt.apply_gradients(zip(
            actor_grad, 
            self.actor.model.trainable_weights
        ))

    def critic_train(
            self, 
            cumul_rewards: tf.Tensor, 
            states: List[Dict[str, np.array]], 
        ):
        """
            Update critic network.

            Parameters:
            -----------
                cumul_rewards: tensorflow.Tensor
                    Cumulative reward.

                states: List[Dict[str, numpy.array]]
                    State list.
        """
        cumul_rewards = np.array(cumul_rewards, dtype=np.float32)
        
        with tf.GradientTape() as tape:
            # Compute the value of the state s
            v = self.critic(states)
            # Calculate the advantage of the state s. cumulative_rewards - value
            advantage = cumul_rewards - v
            # Calculate the loss of the critic network.
            # The loss function is the mean squared error between of the 
            # advantage
            critic_loss = tf.reduce_mean(tf.square(advantage))

        # Compute the gradient of the loss with respect to the critic network
        grad = tape.gradient(critic_loss, self.critic.model.trainable_weights)

        # Apply the gradient to the critic network
        # (See that the apply gradient is used with the optimizer)
        self.critic_opt.apply_gradients(zip(
            grad, 
            self.critic.model.trainable_weights
        ))

    def advantage(
            self, 
            states: List[Dict[str, np.array]], 
            cumul_rewards: tf.Tensor
        ) -> np.array:
        """
            Calculates the advantage given a list of states and a list of 
            cumulative rewards.
        
            Calculate advantage from a state.

            Parameters:
            -----------
                states: List[Dict[str, numpy.array]]
                    State list.

                cumul_rewards: tensorflow.Tensor
                    List of cumulative rewards.

            Return:
            -------
                numpy.array
                    Advantage.
        """
        cumul_rewards = np.array(cumul_rewards, dtype=np.float32)
        advantage = cumul_rewards - self.critic(states)
        return advantage.numpy()

    def update(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            cumul_rewards: tf.Tensor, 
            actor_update_steps: int, 
            critic_update_steps: int
        ):
        """
            Updates the critic and value networks with the constraint of KL 
            divergence. 

            Arguments:
            ----------
                states: List[Dict[str, numpy.array]]
                    State list.

                actions: List[numpy.array] 
                    List of actions corresponding to the states.

                cumul_rewards: tensorflow.Tensor
                    List of cumulative rewards.

                actor_update_steps: int
                    Number of times the actor will be updated.

                critic_update_steps: int
                    Number of times the critic will be updated.
        """
        advantage = self.advantage(states, cumul_rewards)
        self.actor(states)
        old_actor_prob = tf.exp(self.actor.policy_dist.logp(actions))
        old_actor_prob = tf.stop_gradient(old_actor_prob)

        # Update actor
        for _ in range(actor_update_steps):
            self.actor_train(states, actions, advantage, old_actor_prob)

        # Update critic
        for _ in range(critic_update_steps):
            self.critic_train(cumul_rewards, states)

    def get_action(self, state: Dict[str, np.array]) -> np.array:
        """
            Choose an stochastic action.

            Parameters:
            -----------
                state: Dict[str, numpy.array]
                    State.

            Return:
            -------
                numpy.array:
                    Stochastic action.
        """
        return self.actor([state])[0].numpy()

    def get_action_greedy(self, state: Dict[str, np.array]) -> np.array:
        """
            Choose an greedy action.

            Parameters:
            -----------
                state: Dict[str, numpy.array]
                    State.

            Return:
            -------
                numpy.array:
                    Greedy action.
        """
        return self.actor([state], greedy=True)[0].numpy()

    def critic_value(self, state: Dict[str, np.array]) -> tf.Tensor:
        """
            Computes the value of a given state list (Using the critic network).

            Parameters:
            -----------
                state: Dict[str, numpy.array]
                    State.

            Return:
            -------
                tensorflow.Tensor
                    Computed value.
        """
        return self.critic([state])[0, 0]

    def ppo(
            self, 
            env: TeacherEnv, 
            iterations: int=200, 
            max_steps: int=200, 
            save_interval: int=10,
            actor_path: str='models/actor',
            critic_path: str='models/critic',
            gamma: float=0.9, 
            mode: str='train', 
            batch_size: int=32, 
            actor_update_steps: int=10, 
            critic_update_steps: int=10,
            plot_func: Optional[Callable]=None
        ) -> List[float]:
        """
            Full PPO CLIP Algorithm

            Arguments:
            ----------
                env: TeacherEnv
                    Learning environment.

                iterations: int, optional
                    Total number of iterations.
                    Default: 200

                max_steps: int
                    Maximum number of steps for one episode.
                    Default: 200

                save_interval: int, optional
                    Time steps for saving.
                    Default: 10

                actor_path: str, optional
                    Path where the actor model.
                    Default: 'models/actor'

                critic_path: str, optional
                    Path where the critic model.
                    Default: 'models/critic'

                gamma: float, optional
                    Reward discount factor.
                    Default: 0.9

                mode: {'train', 'test'}, optional
                    Train or test mode.
                    Default: 'train'

                batch_size: int, optional
                    Update batch size.
                    Default: 32

                actor_update_steps: int, optional
                    Number of times the actor will be updated.
                    Default: 10

                critic_update_steps: int, optional
                    Number of times the critic will be updated.
                    Default: 10

                plot_func: Optional[Callable], optional
                    Plot function
                    Default: None

            Return:
            -------
                List[float]
                    Cumulative rewards list.
        """
        t0 = time.time()
        t = t0

        # TRAINING MODE.
        if mode == 'train':
            print(
                '\033[1;36m***\033[0m INITIALIZING TRAINING\n' +
                f'Algorithm: {self.name}  | Environment: {env.spec.id}\n'
            )
            cumulative_rewards = []

            # For every iteration
            for it in range(1, iterations + 1):
                # Reset environment 
                state = env.reset()
                buffer_states, buffer_actions, buffer_rewards = [], [], []
                cumulative_it_reward = 0

                # For every episode
                for ep in range(max_steps):  
                    # Get and apply an action.
                    action = self.get_action(state)
                    buffer_states.append(state)
                    buffer_actions.append(action)
                    state, reward, done, _ = env.step(action)

                    # Get environment reward
                    buffer_rewards.append(reward)
                    cumulative_it_reward += float(reward)

                    # Update PPO
                    if (ep + 1)%batch_size == 0 or ep == max_steps - 1 or done:
                        # Get critic value
                        if done: critic_val = 0
                        else: critic_val = self.critic_value(state)

                        # Compute current discount reward
                        discounted_reward = []
                        for r in buffer_rewards[::-1]:
                            critic_val = r + gamma * critic_val
                            discounted_reward.append(critic_val)
                        discounted_reward.reverse()

                        # Update actor and critic parameters
                        self.update(
                            buffer_states, 
                            np.vstack(buffer_actions), 
                            np.array(discounted_reward)[:, np.newaxis], 
                            actor_update_steps, 
                            critic_update_steps
                        )

                        # Clear buffers
                        buffer_states, buffer_actions, buffer_rewards = [], [], []

                    # End episode
                    if done: break

                # Print iteration information
                print(
                    'ITERATION: {it}/{iterations} | ' +
                    'Cumulative reward: {:.4f} | '.format(cumulative_it_reward) +
                    'Iteration time: {:.4f} | '.format(time.time() - t) + 
                    'Cumulative time: {:.4f} | '.format(time.time() - t0)
                )
                t = time()

                cumulative_rewards.append(cumulative_it_reward)

                # Print rewards
                if plot_func is not None:
                    plot_func(cumulative_rewards)
                if it > 0 and it % save_interval == 0:
                    self.save_models(actor_path, critic_path)

            self.save_models(actor_path, critic_path)

        # TEST MODE
        elif mode == 'test':
            self.load_models(actor_path, critic_path)
            print(f'\033[1;36m***\033[0m TESTING PPO.\n')
            cumulative_rewards = []

            # For every iteration
            for it in range(iterations):
                # Reset environment
                cumulative_it_reward = 0
                state = env.reset()

                for _ in range(max_steps):
                    action = self.get_action_greedy(state)
                    state, reward, done, info = env.step(action)
                    cumulative_it_reward += float(reward)
                    if done: break

                # Print iteration information
                print(
                    'ITERATION: {it}/{iterations} | ' +
                    'Cumulative reward: {:.4f} | '.format(cumulative_it_reward) +
                    'Iteration time: {:.4f} | '.format(time.time() - t) + 
                    'Cumulative time: {:.4f} | '.format(time.time() - t0)
                )
                t = time()

            cumulative_rewards.append(cumulative_it_reward)
            if plot_func: plot_func(cumulative_rewards)
        
        else: raise Exception('Unknown mode type: "{mode}"')

        return cumulative_rewards

