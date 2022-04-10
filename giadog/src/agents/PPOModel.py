"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Proximal Policy Optimization (with support for Natural Policy Gradient).

    This imlementation is a modification of the RLZoo implementation of PPO

    References:
    -----------
        * [TODO]
"""
import os
import gym
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from typing import List, Dict
from policy_networks import TeacherANN
from value_networks import TeacherValueANN
from __env__ import EPSILON, LEARNING_RATE


class PPOModel(object):
    """
        This class implements the PPO clip algorithm.
    
        References:
        -----------
            * https://arxiv.org/abs/1707.06347
    """

    def __init__(
            self, 
            action_space: gym.Env,
            observation_space: gym.Env,
            epsilon: float=0.2
        ):
        """
            [TODO]
        """
        self.actor = TeacherANN(action_space, observation_space)
        self.critic = TeacherValueANN(observation_space)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
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

    def __actor_train(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            old_actor_prob: tf.Tensor
        ):
        """
            Update policy (actor) ANN.
            
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

        # Compute the gradient of the loss with respect to the policy ANN
        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_weights
        )
        # Apply the gradient to the policy ANN
        # (See that the apply gradient is used with the optimizer)
        self.actor_opt.apply_gradients(zip(
            actor_grad, 
            self.actor.model.trainable_weights
        ))

    def __critic_train(
            self, 
            cumul_rewards: tf.Tensor, 
            states: List[Dict[str, np.array]], 
        ):
        """
            Update critic ANN.

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
            # Calculate the loss of the critic ANN.
            # The loss function is the mean squared error between of the 
            # advantage
            critic_loss = tf.reduce_mean(tf.square(advantage))

        # Compute the gradient of the loss with respect to the critic ANN
        grad = tape.gradient(critic_loss, self.critic.model.trainable_weights)

        # Apply the gradient to the critic ANN
        # (See that the apply gradient is used with the optimizer)
        self.critic_opt.apply_gradients(zip(
            grad, 
            self.critic.model.trainable_weights
        ))

    def __advantage(
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
            Updates the critic and value ANNs with the constraint of KL 
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
        advantage = self.__advantage(states, cumul_rewards)
        self.actor(states)
        old_actor_prob = tf.exp(self.actor.policy_dist.logp(actions))
        old_actor_prob = tf.stop_gradient(old_actor_prob)

        # Update actor
        for _ in range(actor_update_steps):
            self.__actor_train(states, actions, advantage, old_actor_prob)

        # Update critic
        for _ in range(critic_update_steps):
            self.__critic_train(cumul_rewards, states)

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
            Computes the value of a given state list (Using the critic ANN).

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
