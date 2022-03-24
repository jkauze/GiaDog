"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Trust Region Policy Optimization (with support for Natural Policy Gradient)

    References:
    -----------
        * [TODO]
"""
import copy
import time
import numpy as np
import tensorflow as tf
from src.agents import *
from src.__env__ import EPSILON
from src.training.GiadogGym import TeacherEnv


class TRPO(object):
    """
        [TODO]
    """

    def __init__(
            self, 
            actor: TeacherNetwork,
            critic: TeacherValueNetwork,
            critic_optimizer: tf.keras.optimizers.Optimizer, 
            damping_coeff: float=0.1, 
            cg_iters: int=10, 
            delta: float=0.01
        ):
        """
        Arguments
        ---------
            actor: TeacherNetwork
                Neural network that works as policy.

            critic: TeacherValueNetwork
                Neural network that works as critic.
            
            critic_optimizer: tensorflow.keras.optimizers.Optimizer
                Critic Training Optimizer.

            damping_coeff: float, optional
                Artifact for numerical stability.
                Default: 0.1

            cg_iters: int, optional
                Number of iterations of conjugate gradient to perform.
                Default: 10

            delta: float, optional
                KL-divergence limit for TRPO update.
                Default: 0.01
        """
        self.name = 'TRPO'
        self.actor = actor
        self.critic = critic
        self.critic_opt = critic_optimizer
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.delta = delta
        self.old_dist = DiagGaussian(
            self.actor.action_space.shape[0],
            np.zeros(16,),
            np.ones((16,))
        )

    @staticmethod
    def flat_concat(xs: tf.Tensor) -> tf.Tensor:
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
    def assign_params_from_flat(x: tf.Tensor, params: List[tf.Tensor]):
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
        return self.flat_concat(self.actor.model.trainable_weights)

    def set_actor_params(self, params: tf.Tensor):
        """
            Set actor trainable parameters
            
            Parameters:
            -----------
                params: tensorflow.Tensor:
                    Parameters to set.
        """
        self.assign_params_from_flat(
            params, 
            self.actor.model.trainable_weights
        )

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

    def advantage(
            self, 
            states: List[Dict[str, np.array]], 
            cumul_reward: tf.Tensor
        ) -> np.array:
        """
            Calculate advantage from a state.

            Parameters:
            -----------
                states: List[Dict[str, numpy.array]]
                    State list.

                cumul_reward: tensorflow.Tensor
                    Cumulative reward.

            Return:
                numpy.array
                    Advantage.
        """
        cumul_reward = np.array(cumul_reward, dtype=np.float32)
        adv = cumul_reward - self.critic(states)
        return adv.numpy()

    def critic_value(self, state: Dict[str, np.array]) -> tf.Tensor:
        """
            Compute value from critic.
            
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

    def critic_train(
            self, 
            cumul_reward: tf.Tensor, 
            states: List[Dict[str, np.array]], 
        ):
        """
            Update critic network.

            Parameters:
            -----------
                cumul_reward: tensorflow.Tensor
                    Cumulative reward.

                states: List[Dict[str, numpy.array]]
                    State list.
        """
        # Get critic loss
        cumul_reward = np.array(cumul_reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            v = self.critic(states)
            advantage = cumul_reward - v
            critic_loss = tf.reduce_mean(tf.square(advantage))

        # Compute gradient and update critic
        grad = tape.gradient(critic_loss, self.critic.model.trainable_weights)
        self.critic_opt.apply_gradients(zip(
            grad, 
            self.critic.model.trainable_weights
        ))

    def eval(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array],  
            advantage: np.array, 
            oldpi_prob: tf.Tensor
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

                oldpi_prob: tensorflow.Tensor
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
        pi_prob = tf.exp(self.actor.policy_dist.logp(actions))
        ratio = pi_prob / (oldpi_prob + EPSILON)

        # Objetive function
        surr = ratio * advantage
        actor_loss = -tf.reduce_mean(surr)

        # Kullback-Leibler Divergence
        kl = self.old_dist.kl(*self.actor.policy_dist.get_param())
        kl = tf.reduce_mean(kl)

        return actor_loss, kl

    def hessian_vector_product(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            oldpi_prob: tf.Tensor,
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

                oldpi_prob: tensorflow.Tensor
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
                _, kl = self.eval(states, actions, advantage, oldpi_prob)
            # Kullback-Leibler Divergence Gradient (Gradient DKL)
            g = tape0.gradient(kl, params)

            g = self.flat_concat(g)
            # Gradient DKL * x
            v = tf.reduce_sum(g * v_ph)
        # Gradient (Gradient DKL * x)
        grad = tape1.gradient(v, params)

        hvp = self.flat_concat(grad)
        if self.damping_coeff > 0: hvp += self.damping_coeff * v_ph
        return hvp

    def conjugate_gradient(self, Ax: Callable, b: tf.Tensor) -> tf.Tensor:
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

    def actor_train(
            self, 
            states: List[Dict[str, np.array]], 
            actions: List[np.array], 
            advantage: np.array, 
            oldpi_prob: tf.Tensor, 
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

                oldpi_prob: tensorflow.Tensor
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
            actor_loss, kl = self.eval(states, actions, advantage, oldpi_prob)
        actor_grad = self.flat_concat(tape.gradient(
            actor_loss, 
            self.actor.model.trainable_weights
        ))

        # Compute the conjugate gradient to obtain X_k = H_k^-1 . g_k
        Hx = lambda x: self.hessian_vector_product(
            states, 
            actions, 
            advantage, 
            oldpi_prob, 
            x
        )
        x = self.conjugate_gradient(Hx, actor_grad)
        
        # Compute estimated propose step
        alpha = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + EPSILON))

        # Linear search for TRPO
        old_params = self.flat_concat(self.actor.model.trainable_weights)
        for j in range(backtrack_iters):
            # Compute proposed update
            self.set_actor_params(old_params - alpha * x * backtrack_coeff ** j)

            # Obtain news actor objetive function value and the Kullback-Leibler 
            # Divergence
            kl, new_actor_loss = self.eval(states, actions, advantage, oldpi_prob)

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
            cumul_reward: tf.Tensor, 
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

                cumul_reward: tensorflow.Tensor
                    Cumulative reward.

                train_critic_iters: int
                    Number of times the critic will be updated.

                backtrack_iters: int
                    Number of iterations for the Linear Search in TRPO.

                backtrack_coeff: float
                    Decrease coefficient for the Linear Search in TRPO.
        """
        adv = self.advantage(states, cumul_reward)
        _ = self.actor(states)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(actions))
        oldpi_prob = tf.stop_gradient(oldpi_prob)
        oldpi_param = self.actor.policy_dist.get_param()
        self.old_dist.set_param(*oldpi_param)

        self.actor_train(
            states, 
            actions, 
            adv, 
            oldpi_prob, 
            backtrack_iters,
            backtrack_coeff
        )

        for _ in range(train_critic_iters):
            self.critic_train(cumul_reward, states)

    def trpo(
            self, 
            env: TeacherEnv, 
            iterations: int=200, 
            max_steps: int=200, 
            save_interval: int=10,
            actor_path: str='models/actor',
            critic_path: str='models/critic',
            gamma: float=0.9, 
            mode: str='train', 
            render: bool=False, 
            batch_size: int=32, 
            backtrack_iters: int=10, 
            backtrack_coeff: float=0.8,
            train_critic_iters: int=80, 
            plot_func: Optional[Callable]=None
        ) -> List[float]:
        """
            Full TRPO Algorithm

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

                render: bool, optional
                    Render each step.
                    Default: False

                batch_size: int, optional
                    Update batch size.
                    Default: 32

                backtrack_iters: int, optional
                    Maximum number of steps allowed in the backtracking line 
                    search.
                    Default: 10

                backtrack_coeff: float, optional
                    How far back to step during backtracking line search.
                    Default: 0.8

                train_critic_iters: int, optional
                    Critic update iteration steps.
                    Default: 80

                plot_func: Optional[Callable], optional
                    Plot function
                    Default: None

            Return:
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

                # For every ep-th episode
                for ep in range(max_steps):  
                    # Get and apply an action.
                    action = self.get_action(state)
                    buffer_states.append(state)
                    buffer_actions.append(action)
                    state, reward, done, _ = env.step(action)

                    # Get environment reward
                    buffer_rewards.append(reward)
                    cumulative_it_reward += reward

                    # Update TRPO
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
                            buffer_actions, 
                            np.array(discounted_reward)[:, np.newaxis], 
                            train_critic_iters, 
                            backtrack_iters, 
                            backtrack_coeff
                        )

                        # Clear buffers
                        buffer_states, buffer_actions, buffer_rewards = [], [], []

                    # End episode
                    if done: break

                # Print iteration information
                print(
                    'ITERATION: {it}/{iterations} | ' +
                    'Cumulative reward: {:.4f}'.format(cumulative_it_reward) +
                    'Iteration time: {:.4f}'.format(time.time() - t) + 
                    'Cumulative time: {:.4f}'.format(time.time() - t0)
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
            print(f'Testing TRPO Algorithm.')
            cumulative_rewards = []

            # For every iteration
            for it in range(iterations):
                # Reset environment
                cumulative_it_reward = 0
                state = env.reset()

                for _ in range(max_steps):
                    action = self.get_action_greedy(state)
                    state, reward, done, _ = env.step(action)
                    cumulative_it_reward += reward
                    if done: break

                # Print iteration information
                print(
                    'ITERATION: {it}/{iterations} | ' +
                    'Cumulative reward: {:.4f}'.format(cumulative_it_reward) +
                    'Iteration time: {:.4f}'.format(time.time() - t) + 
                    'Cumulative time: {:.4f}'.format(time.time() - t0)
                )
                t = time()


            cumulative_rewards.append(cumulative_it_reward)
            if plot_func: plot_func(cumulative_rewards)

        else: raise Exception('Unknown mode type: "{mode}"')

        return cumulative_rewards
