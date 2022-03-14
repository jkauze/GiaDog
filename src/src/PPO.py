"""
Authors: Amin Arriaga, Eduardo Lopez
Project: Graduation Thesis: GIAdog

This file contians the implementation of the PPO algorithm (PPO clip).

This imlementation is a modification of the RLZoo implementation of PPO.

References:
-----------
https://github.com/tensorlayer/RLzoo/blob/master/rlzoo/algorithms/ppo_clip/ppo_clip.py

"""
import numpy as np
import tensorflow as tf
import time

#from rlzoo.common.utils import *
#from rlzoo.common.policy_networks import *
#from rlzoo.common.value_networks import *


EPS = 1e-8  # epsilon (Epsilon parameter to avoid zero division)


###############################  PPO  ####################################

class PPO_CLIP(object):
    """
    PPO class. This class implements the PPO clip algorithm.
    For more information about the PPO algorithm, please refer to the paper:
    https://arxiv.org/abs/1707.06347
    """

    def __init__(self, net_list, optimizers_list, epsilon=0.2):
        """
        Initialize PPO class
        
        Arguments:
        ---------
        net_list:  A list of networks (value and policy(actor)) used in the 
                    algorithm, from common functions or customization.
        optimizers_list: a list of optimizers for all networks and 
                                differentiable variables
        epsilon: clip parameter
        
        Returns:
        -------
        None

        """
        assert len(net_list) == 2
        assert len(optimizers_list) == 2

        self.name = 'PPO_CLIP'
        self.epsilon = epsilon

        self.critic, self.actor = net_list

        #assert isinstance(self.critic, ValueNetwork)
        #assert isinstance(self.actor, StochasticPolicyNetwork)

        self.critic_opt, self.actor_opt = optimizers_list

    def a_train(self, tfs, tfa, tfadv, oldpi_prob):
        """
        Update policy (actor) network
        
        Arguments:
        ---------

        tfs: state
        tfa: action
        tfadv: advantage
        oldpi_prob: old policy distribution
        
        Returns:
        -------
        None

        """
        try:
            tfs   = np.array(tfs, np.float32)
            tfa   = np.array(tfa, np.float32)
            tfadv = np.array(tfadv, np.float32)
        except:
            pass

        with tf.GradientTape() as tape:
            _ = self.actor(tfs)
            # We compute the probability of the actions taken by the actor
            pi_prob = tf.exp(self.actor.policy_dist.logp(tfa))
            print(pi_prob)
            # Calculate the ratio between the old and the new policy
            ratio = pi_prob / (oldpi_prob + EPS)

            # Calculate the surrogate loss. pi_new/pi_old * advantage
            surr = ratio * tfadv
            
            # Apply the clipped surrogate function (check the PPO clip update) 
            aloss = -tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1.+\
                            self.epsilon) * tfadv))
        # Compute the gradient of the loss with respect to the policy network
        a_gard = tape.gradient(aloss, self.actor.model.trainable_weights)
        #a_grad_sigma = tape.gradient(aloss, self.actor.log_std)
        # Apply the gradient to the policy network
        # (See that the apply gradient is used with the optimizer)
        self.actor_opt.apply_gradients(zip(a_gard, self.actor.model.trainable_weights))
        #self.actor_opt.apply_gradients(zip(a_grad_sigma, self.actor.log_std))

    def c_train(self, tfdc_r, s):
        """
        Update critic network
        
        Arguments:
        ---------
        :param tfdc_r: cumulative reward
        :param s: state
        
        Returns:
        -------
        None
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        
        with tf.GradientTape() as tape:
            # Compute the value of the state s
            v = self.critic(s)
            # Calculate the advantage of the state s. cumulative_rewards - value
            advantage = tfdc_r - v
            # Calculate the loss of the critic network.
            # The loss function is the mean squared error between of the 
            # advantage
            closs = tf.reduce_mean(tf.square(advantage))
        # Compute the gradient of the loss with respect to the critic network
        grad = tape.gradient(closs, self.critic.model.trainable_weights)
        # Apply the gradient to the critic network
        # (See that the apply gradient is used with the optimizer)
        self.critic_opt.apply_gradients(zip(grad, self.critic.model.trainable_weights))

    def cal_adv(self, tfs, tfdc_r):
        """
        Calculates advantage
        
        Arguments:
        ---------
        :param tfs: state
        :param tfdc_r: cumulative reward

        Returns:
        -------
        advantage: advantage value
        
        """
        tfdc_r = np.array(tfdc_r, dtype=np.float32)
        advantage = tfdc_r - self.critic(tfs)
        return advantage.numpy()

    def update(self, s, a, r, a_update_steps, c_update_steps):
        """
        Update parameter with the constraint of KL divergent/

        Arguments:
        ---------
        :param s: state
        :param a: act
        :param r: reward
        
        Returns:
        -------
        None
        """
        adv = self.cal_adv(s, r)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)  # adv norm, sometimes helpful

        _ = self.actor(s)
        oldpi_prob = tf.exp(self.actor.policy_dist.logp(a))
        oldpi_prob = tf.stop_gradient(oldpi_prob)

        # update actor
        for _ in range(a_update_steps):
            self.a_train(s, a, adv, oldpi_prob)

        # update critic
        for _ in range(c_update_steps):
            self.c_train(r, s)

    def get_action(self, s):
        """
        Compute the agent action given an state s.

        Arguments:
        ---------
        s: state

        Returns:
        -------
        clipped action
        """

        return self.actor(s)[0].numpy()

    def get_action_greedy(self, s):
        """
        Compute the agent action given an state s, based on a greedy policy.

        Arguments:
        ---------
        s: state

        Returns:
        -------
        clipped action
        """
        return self.actor(s, greedy=True)[0].numpy()

    def get_value(self, s):
        """
        Compute the value of a given state (Using the critic network).

        Arguments:
        ---------
        s: state

        Returns:
        -------
        value
        """
        try:
            s = s.astype(np.float32)
            if s.ndim < 2: s = s[np.newaxis, :]
        except:
            pass
        res = self.critic(s)[0, 0]
        return res

    """
    def save_ckpt(self, env_name):
    """
    """save trained weights
    :return: None"""
    """
    save_model(self.actor, 'actor', self.name, env_name)
    save_model(self.critic, 'critic', self.name, env_name)

    def load_ckpt(self, env_name):
    """
    """load trained weights
    :return: None"""
    """
    load_model(self.actor, 'actor', self.name, env_name)
    load_model(self.critic, 'critic', self.name, env_name)
    """
    def learn(self, env, train_episodes=200, test_episodes=100, max_steps=200, save_interval=10,
              gamma=0.9, mode='train', render=False, batch_size=32, a_update_steps=10, c_update_steps=10,
              plot_func=None):
        """
        learn function
        :param env: learning environment
        :param train_episodes: total number of episodes for training
        :param test_episodes: total number of episodes for testing
        :param max_steps: maximum number of steps for one episode
        :param save_interval: timesteps for saving
        :param gamma: reward discount factor
        :param mode: train or test
        :param render: render each step
        :param batch_size: udpate batchsize
        :param a_update_steps: actor update iteration steps
        :param c_update_steps: critic update iteration steps
        :param plot_func: additional function for interactive module
        :return: None
        """

        t0 = time.time()

        if mode == 'train':
            print('Training...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for ep in range(1, train_episodes + 1):
                s = env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_rs_sum = 0
                for t in range(max_steps):  # in one episode
                    if render:
                        env.render()
                    a = self.get_action(s)

                    s_, r, done, _ = env.step(a)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)
                    s = s_
                    ep_rs_sum += r

                    # update ppo
                    if (t + 1) % batch_size == 0 or t == max_steps - 1 or done:
                        if done:
                            v_s_ = 0
                        else:
                            try:
                                v_s_ = self.get_v(s_)
                            except:
                                v_s_ = self.get_v([s_])   # for raw-pixel input

                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + gamma * v_s_
                            discounted_r.append(v_s_)
                        discounted_r.reverse()
                        # bs = buffer_s if len(buffer_s[0].shape)>1 else np.vstack(buffer_s) # no vstack for raw-pixel input
                        bs = buffer_s
                        ba, br = np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                        buffer_s, buffer_a, buffer_r = [], [], []

                        self.update(bs, ba, br, a_update_steps, c_update_steps)
                    if done:
                        break

                print(
                    'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                        ep, train_episodes, ep_rs_sum,
                        time.time() - t0
                    )
                )

                reward_buffer.append(ep_rs_sum)
                if plot_func is not None:
                    plot_func(reward_buffer)
                if ep and not ep % save_interval:
                    self.save_ckpt(env_name=env.spec.id)
                    plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

            self.save_ckpt(env_name=env.spec.id)
            plot_save_log(reward_buffer, algorithm_name=self.name, env_name=env.spec.id)

        # test
        elif mode == 'test':
            self.load_ckpt(env_name=env.spec.id)
            print('Testing...  | Algorithm: {}  | Environment: {}'.format(self.name, env.spec.id))
            reward_buffer = []
            for eps in range(test_episodes):
                ep_rs_sum = 0
                s = env.reset()
                for step in range(max_steps):
                    if render:
                        env.render()
                    action = self.get_action_greedy(s)
                    s, reward, done, info = env.step(action)
                    ep_rs_sum += reward
                    if done:
                        break

                print('Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    eps, test_episodes, ep_rs_sum, time.time() - t0)
                )
                reward_buffer.append(ep_rs_sum)
                if plot_func:
                    plot_func(reward_buffer)
        else:
            print('unknown mode type')