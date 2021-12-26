"""
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/12/24

	Description Gym enviroment for the training of the control policies
"""

from typing import *

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding

from simulation import Simulation 

from pkg_resources import parse_version

class OpenBlackyEnv(gym.Env):
	"""
	Description:
		The agent (a quadrupedal robot) is started at a random position in the terrain. 
		For a given state the agent would set the desired robot joint configuration.
	
	Source:
		An early version of this envriment first appeared on the article:
		
		Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
		(p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
		https://arxiv.org/pdf/2010.11251.pdf
	
	Observation:
		Type: Box(2)
		(TODO)
	Actions:
		Type: Continuous(12)
		(TODO)
	Reward:
		(TODO)
	Starting State:
		(TODO)
	Episode Termination:
		(TODO)
	"""
	def __init__(self, goal_velocity=0):
		self.min_position = -1.2
		self.max_position = 0.6
		self.max_speed = 0.07
		self.goal_position = 0.5
		self.goal_velocity = goal_velocity

		self.force = 0.001
		self.gravity = 9.807

		self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
		self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

		
		self.viewer = None


		# The action space is fixed -> The agent can only control the robot 12 motors
		self.action_space = spaces.Box(low= -np.inf, high=np.inf, shape=(12,))
		
		# The observation_space should be changed depending on the mode: priviledged
		# or real world simulation 

		if self.mode == "teacher":

			self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
		
		elif self.mode == "student":

			self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
		




	def step(self, action):
		"""
		
		Should return :
		state, reward, done, and an empty dictionary {}?

		* It is preferabel to configure the step function such that the sensor 
		actualization times macht tge real ones
		"""
		assert self.action_space.contains(
			action
		), f"{action!r} ({type(action)}) invalid"

		pass
		#return np.array(self.state, dtype=np.float32), reward, done, {}

	def reset(self, seed: Optional[int] = None):
		"""
		TODO
		
		Should return state 
		"""
		
		pass


	def render(self, mode="human"):
		"""
		TODO
		"""
		pass

	def reward(self):
		"""
		TODO
		"""
		pass

	def termination(self):
		"""
		TODO
		
		Conditions for the episode to end
		"""
		pass
	
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]
	
	def reward(self):
		"""
		TODO
		"""
		pass

	def get_observation(self):
		"""
		TODO
		"""
		pass
	
	def noisy_observation(self):
		"""
		TODO

		Modify the code below to match our system.

		* Change the function only to affect interest observations depending on 
		the enviroment
	
		"""
		
		self._get_observation()
		observation = np.array(self._observation)
		if self._observation_noise_stdev > 0:
			observation += (self.np_random.normal(
				scale=self._observation_noise_stdev, size=observation.shape) *
							self.minitaur.GetObservationUpperBound())
		return observation
	

