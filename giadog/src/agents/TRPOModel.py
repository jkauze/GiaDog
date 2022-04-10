"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Trust Region Policy Optimization (with support for Natural Policy Gradient).

    This imlementation is a modification of the RLZoo implementation of PPO

    References:
    -----------
        * [TODO]
"""
import os
import gym
import copy
import numpy as np
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from distributions import DiagGaussian
from policy_networks import TeacherANN
from value_networks import TeacherValueANN
from __env__ import EPSILON, LEARNING_RATE
from typing import List, Dict, Tuple, Callable


class TRPOModel(object):
    """
        [TODO]
    """
    def __init__(
            self, 
            action_space: gym.Env,
            observation_space: gym.Env,
            damping_coeff: float=0.1, 
            cg_iters: int=10, 
            delta: float=0.01
        ):
        """
            [TODO]
        """
        self.actor = TeacherANN(action_space, observation_space)
        self.critic = TeacherValueANN(observation_space)
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.delta = delta
        self.old_dist = DiagGaussian(
            self.actor.action_space.shape[0],
            np.zeros(16,),
            np.ones((16,))
        )

    @staticmethod
    def __flat_concat(xs: tf.Tensor) -> tf.Tensor:
        """
            Flat concat input.

            Arguments:
            ----------
                xs: tensorflow.Tensor
                    A list of tensor

            Return:
            -------
                tensorflow.Tensor
                    Flat tensor
        """
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    @staticmethod
    def __assign_params_from_flat(x: tf.Tensor, params: List[tf.Tensor]):
        """
            Assign params from flat input.

            Arguments:
            ----------
                x: tensorflow.Tensor
                    Flat tensor.

                params: List[tensorflow.Tensor]
                    Params to assign.
        """
        # The 'int' is important for scalars
        flat_size = lambda p: int(np.prod(p.shape.as_list()))  
        splits = tf.split(x, [flat_size(p) for p in params])
        new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
        return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])

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

    def get_actor_params(self) -> tf.Tensor:
        """
            Get actor trainable parameters

            Return:
            -------
                tensorflow.Tensor:
                    Flat actor trainable parameters
        """
        return self.__flat_concat(self.actor.model.trainable_weights)

    def set_actor_params(self, params: tf.Tensor):
        """
            Set actor trainable parameters
            
            Parameters:
            -----------
                params: tensorflow.Tensor:
                    Parameters to set.
        """
        self.__assign_params_from_flat(
            params, 
            self.actor.model.trainable_weights
        )

    def __advantage(
            self, 
            states: List[Dict[str, np.array]], 
            cumul_rewards: tf.Tensor
        ) -> np.array:
        """
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
                    List of cumulative rewards.

                states: List[Dict[str, numpy.array]]
                    State list.
        """
        # Get critic loss
        cumul_rewards = np.array(cumul_rewards, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(states)
            advantage = cumul_rewards - v
            critic_loss = tf.reduce_mean(tf.square(advantage))

        # Compute gradient and update critic
        grad = tape.gradient(critic_loss, self.critic.model.trainable_weights)
        self.critic_opt.apply_gradients(zip(
            grad, 
            self.critic.model.trainable_weights
        ))

    def __eval(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array],  
            advantage: np.array, 
            old_actor_prob: tf.Tensor
        ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
            Objective Function.

            Parameters:
            -----------
                states: List[Dict[str, numpy.array]]
                    State list.

                actions: List[numpy.array] 
                    List of actions corresponding to the states.

                advantage: numpy.array 
                    List of advantage obtained in each state.

                old_actor_prob: tensorflow.Tensor
                    Probability obtained by the previous policy.

            Returns:
            --------
                tensorflow.Tensor
                    Actor objetive function value

                tensorflow.Tensor
                    Kullback-Leibler Divergence
        """
        # Calculate the ratio between the current policy and the previous one
        self.actor(states)
        actor_prob = tf.exp(self.actor.policy_dist.logp(actions))
        ratio = actor_prob / (old_actor_prob + EPSILON)

        # Objetive function
        surr = ratio * advantage
        actor_loss = -tf.reduce_mean(surr)

        # Kullback-Leibler Divergence
        kl = self.old_dist.kl(*self.actor.policy_dist.get_param())
        kl = tf.reduce_mean(kl)

        return actor_loss, kl

    def __hessian_vector_product(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            old_actor_prob: tf.Tensor,
            v_ph: tf.Tensor
        ) -> tf.Tensor:
        """
            Calculation of the inverse of the Hessian.

            Arguments:
            ----------
                states: List[Dict[str, numpy.array]]
                    State list.

                actions: List[numpy.array] 
                    List of actions corresponding to the states.

                advantage: numpy.array 
                    List of advantage obtained in each state.

                old_actor_prob: tensorflow.Tensor
                    Probability obtained by the previous policy.

                v_ph: tensorflow.Tensor
                    Vector to be multiplied

            Return:
            -------
                tensorflow.Tensor
                    Hessian inverse.
        """
        params = self.actor.model.trainable_weights

        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                _, kl = self.__eval(states, actions, advantage, old_actor_prob)
            # Kullback-Leibler Divergence Gradient (Gradient DKL)
            g = tape0.gradient(kl, params)

            g = self.__flat_concat(g)
            # Gradient DKL * x
            v = tf.reduce_sum(g * v_ph)
        # Gradient (Gradient DKL * x)
        grad = tape1.gradient(v, params)

        hvp = self.__flat_concat(grad)
        if self.damping_coeff > 0: hvp += self.damping_coeff * v_ph
        return hvp

    def __conjugate_gradient(self, Ax: Callable, b: tf.Tensor) -> tf.Tensor:
        """
            Conjugate gradient algorithm.

            References:
            -----------
                * https://en.wikipedia.org/wiki/Conjugate_gradient_method
        """
        # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. 
        # Change if doing warm start.
        r = copy.deepcopy(b)
          
        x = np.zeros_like(b)
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)

        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPSILON)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new

        return x

    def __actor_train(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            old_actor_prob: tf.Tensor, 
            backtrack_iters: int, 
            backtrack_coeff: float
        ):
        """
            Arguments:
            ----------
                states: List[Dict[str, numpy.array]]
                    State list.

                actions: List[numpy.array] 
                    List of actions corresponding to the states.

                advantage: numpy.array 
                    List of advantage obtained in each state.

                old_actor_prob: tensorflow.Tensor
                    Probability obtained by the previous policy.

                backtrack_iters: int
                    Number of iterations for the Linear Search in TRPO.

                backtrack_coeff: float
                    Decrease coefficient for the Linear Search in TRPO.
        """
        actions = np.array(actions, np.float32)
        advantage = np.array(advantage, np.float32)

        # Compute the actor objetive function value and the Kullback-Leibler 
        # Divergence
        with tf.GradientTape() as tape:
            actor_loss, kl = self.__eval(states, actions, advantage, old_actor_prob)
        actor_grad = self.__flat_concat(tape.gradient(
            actor_loss, 
            self.actor.model.trainable_weights
        ))

        # Compute the conjugate gradient to obtain X_k = H_k^-1 . g_k
        Hx = lambda x: self.__hessian_vector_product(
            states, 
            actions, 
            advantage, 
            old_actor_prob, 
            x
        )
        x = self.__conjugate_gradient(Hx, actor_grad)
        
        # Compute estimated propose step
        alpha = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + EPSILON))

        # Linear search for TRPO
        old_params = self.__flat_concat(self.actor.model.trainable_weights)
        for j in range(backtrack_iters):
            # Compute proposed update
            self.set_actor_params(old_params - alpha * x * backtrack_coeff ** j)

            # Obtain news actor objetive function value and the Kullback-Leibler 
            # Divergence
            kl, new_actor_loss = self.__eval(states, actions, advantage, old_actor_prob)

            # Check the constraint that the divergence is less than a delta 
            # and that the value of the function is less than the previous one.
            # If so, accept new params at step j of line search.
            if kl <= self.delta and new_actor_loss <= actor_loss: break

            # If we reach the last verification, it means that the search 
            # failed and we must return to the previous parameters.
            if j == backtrack_iters - 1:  self.set_actor_params(old_params)

    def update(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            cumul_rewards: tf.Tensor, 
            train_critic_iters: int, 
            backtrack_iters: int, 
            backtrack_coeff: float
        ):
        """
            Update TRPO parameters.

            Arguments:
            ----------
                states: List[Dict[str, numpy.array]]
                    State list.

                actions: List[numpy.array] 
                    List of actions corresponding to the states.

                cumul_rewards: tensorflow.Tensor
                    List of cumulative rewards.

                train_critic_iters: int
                    Number of times the critic will be updated.

                backtrack_iters: int
                    Number of iterations for the Linear Search in TRPO.

                backtrack_coeff: float
                    Decrease coefficient for the Linear Search in TRPO.
        """
        adv = self.__advantage(states, cumul_rewards)
        self.actor(states)
        old_actor_prob = tf.exp(self.actor.policy_dist.logp(actions))
        old_actor_prob = tf.stop_gradient(old_actor_prob)
        oldpi_param = self.actor.policy_dist.get_param()
        self.old_dist.set_param(*oldpi_param)

        self.__actor_train(
            states, 
            actions, 
            adv, 
            old_actor_prob, 
            backtrack_iters,
            backtrack_coeff
        )

        for _ in range(train_critic_iters):
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

