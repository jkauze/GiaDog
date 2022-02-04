"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Description Gym enviroment for the training of the control policies
"""

# Utilities
import numpy as np
from typing import *

# OpenIA Gym
import gym
from gym import spaces

# Simulation
import rospy
import message_filters
from spot_mini_ros.msg import joint_angles, normal_data, priviliged_data, text

class teacher_giadog_env(gym.Env):
    """
        Description:
        ------------
            The agent (a quadrupedal robot) is started at a random position in the 
            terrain. For a given state the agent would set the desired robot joint 
            configuration.
        
        Source:
        -------
            An early version of this envriment first appeared on the article:

            Learning Quadrupedal Locomotion over Challenging Terrain (Oct, 2020).
            (p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
            https://arxiv.org/pdf/2010.11251.pdf
        
        Observation:
        ------------
            (TODO)
        Actions:
        --------
            (TODO)
        Reward:
        -------
            (TODO)
        Starting State:
        ---------------
            (TODO)
        Episode Termination:
        --------------------
            (TODO)
    """
    QUEUE_SIZE = 10
    # ROS publisher node that update the spot mini joints
    reset_pub = rospy.Publisher('reset_pub', text, QUEUE_SIZE)

    def __init__(self):
        self.observation_space = spaces.Dict({
            # Non-priviliged Space
            'gravity_vector': spaces.Box(
                low = -20 * np.ones((3,)), 
                high = np.zeros((3,)),
                dtype = np.float16
            ),
            'angular_vel': spaces.Box(
                low = -5 * np.ones((3,)), 
                high = 5 * np.ones((3,)),
                dtype = np.float16
            ),
            'linear_vel': spaces.Box(
                low = -5 * np.ones((3,)), 
                high = 5 * np.ones((3,)),
                dtype = np.float16
            ),
            'joint_angles': spaces.Box(
                low = np.zeros((12,)), 
                high = 2 * np.pi * np.ones((12,)),
                dtype = np.float16
            ),
            'joint_vels': spaces.Box(
                low = -5 * np.pi * np.ones((3,)), 
                high = 5 * np.pi * np.ones((3,)),
                dtype = np.float16
            ),
            'toes_contact': spaces.MultiBinary(4),
            'thighs_contact': spaces.MultiBinary(4),
            'shanks_contact': spaces.MultiBinary(4),

            # Priviliged Space
            'normal_foot': spaces.Box(
                low = -10 * np.ones((4, 3)), 
                high = 10 * np.ones((4, 3)),
                dtype = np.float16
            ), 
            'height_scan': spaces.Box(
                low = -50 * np.ones((4, 9)), 
                high = 50 * np.ones((4, 9)),
                dtype = np.float16
            ), 
            'foot_forces': spaces.Box(
                low = -10 * np.ones((4,)), 
                high = 10 * np.ones((4,)),
                dtype = np.float16
            ), 
            'foot_friction': spaces.Box(
                low = -2 * np.ones((4,)), 
                high = 2 * np.ones((4,)),
                dtype = np.float16
            )
        })
        self.action_space = spaces.Box(
            low = -2 * np.pi * np.ones((4, 16)),
            high = 2 * np.pi * np.ones((4, 16)),
            dtype = np.float16
        )

    def step(self, action: np.ndarray):
        pass

    def reset(self, terrain_file: str=''):
        # Create message
        msg = text()
        msg.txt= terrain_file
        self.reset_pub.publish(msg)

    def close(self):
        pass
