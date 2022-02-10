"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Description Gym enviroment for the training of the control policies
"""

# Utilities
import numpy as np
from typing import *
from random import randint

# OpenIA Gym
import gym
from gym import spaces

# Simulation
import rospy
import message_filters
from src.terrain_gen import *
from spot_mini_ros.msg import joint_angles, normal_data, priviliged_data, text, timestep

# Controller
#from src.neural_networks import *
from src.inverse_kinematics import *
from src.foot_trajectory_generator import *

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
    VEL_TH   = 0.6 # Velocity threshold
    SWIGN_PH = 0   # Swign phase
    TERRAIN_FILE = 'gym_terrain.txt'
    HISTORY_LEN = 100
    FOOT_HISTORY_LEN = 3

    # ROS publisher node that update the spot mini joints
    reset_pub = rospy.Publisher('reset_simulation', text, queue_size=QUEUE_SIZE)
    # ROS publisher node that update the spot mini joints
    joints_pub = rospy.Publisher('spot_joints', joint_angles, queue_size=QUEUE_SIZE)

    def __init__(self):
        self.observation_space = spaces.Dict({
            # Non-priviliged Space
            'gravity_vector': spaces.Box(
                low = -20 * np.ones((3,)), 
                high = np.zeros((3,)),
                dtype = np.float32
            ),
            'angular_vel': spaces.Box(
                low = -5 * np.ones((3,)), 
                high = 5 * np.ones((3,)),
                dtype = np.float32
            ),
            'linear_vel': spaces.Box(
                low = -5 * np.ones((3,)), 
                high = 5 * np.ones((3,)),
                dtype = np.float32
            ),
            'joint_angles': spaces.Box(
                low = np.zeros((12,)), 
                high = 2 * np.pi * np.ones((12,)),
                dtype = np.float32
            ),
            'joint_vels': spaces.Box(
                low = -5 * np.pi * np.ones((3,)), 
                high = 5 * np.pi * np.ones((3,)),
                dtype = np.float32
            ),
            'toes_contact': spaces.MultiBinary(4),
            'thighs_contact': spaces.MultiBinary(4),
            'shanks_contact': spaces.MultiBinary(4),

            # Priviliged Space
            'normal_foot': spaces.Box(
                low = -10 * np.ones((4, 3)), 
                high = 10 * np.ones((4, 3)),
                dtype = np.float32
            ), 
            'height_scan': spaces.Box(
                low = -50 * np.ones((4, 9)), 
                high = 50 * np.ones((4, 9)),
                dtype = np.float32
            ), 
            'foot_forces': spaces.Box(
                low = -10 * np.ones((4,)), 
                high = 10 * np.ones((4,)),
                dtype = np.float32
            ), 
            'foot_friction': spaces.Box(
                low = -2 * np.ones((4,)), 
                high = 2 * np.ones((4,)),
                dtype = np.float32
            )
        })
        self.action_space = spaces.Box(
            low = -float('inf') * np.ones((16,)),
            high = float('inf') * np.ones((16,)),
            dtype = np.float32
        )
        self.gravity_vector = np.array([0, 0, -9.807])

        normal_data_sub = message_filters.Subscriber(
            'normal_data', 
            normal_data
        )
        priviliged_data_sub = message_filters.Subscriber(
            'priviliged_data', 
            priviliged_data
        )
        timestep_sub = message_filters.Subscriber(
            'timestep',
            timestep
        )
        ts = message_filters.ApproximateTimeSynchronizer(
            [timestep_sub, normal_data_sub, priviliged_data_sub], 
            queue_size=self.QUEUE_SIZE,
            slop=0.1,
            allow_headerless=True
        )
        ts.registerCallback(self.__update_obs)

        #self.H = np.zeros((self.HISTORY_LEN, controller_neural_network.NORMAL_DATA_SHAPE))
        self.foot_target_hist = np.zeros((self.FOOT_HISTORY_LEN, 12))
        #self.model = teacher_nn()
    
    def __actuate_joints(cls, joint_target_positions: List[float]):
        """
            Moves the robot joints to a given target position.

            Argument:
            ---------
                joint_target_positions: List[float], shape (12,)
                    Quadruped joints desired angles. 
                    The order is the same as for the robot actuated_joints_ids.
                    The order should be as follows:
                        'motor_front_left_hip' 
                        'motor_front_left_upper_leg'// "Front left thigh"
                        'motor_front_left_lower_leg'// "Front left shank"

                        'motor_front_right_hip' 
                        'motor_front_right_upper_leg'// "Front right thigh"
                        'motor_front_right_lower_leg'// "Front right shank"

                        'motor_back_left_hip' 
                        'motor_back_left_upper_leg'// "Back left thigh"
                        'motor_back_left_lower_leg'// "Back left shank"

                        'motor_back_right_hip' 
                        'motor_back_right_upper_leg'// "Back right thigh"
                        'motor_back_right_lower_leg'// "Back right shank"
        """
        # Create message
        msg = joint_angles()
        msg.joint_target_positions = joint_target_positions
        cls.joints_pub.publish(msg)

    def __get_reward(self, ftg_freqs: List[float]) -> float:
        """
            Reward function.

            Arguments:
            ----------
                ftg_freqs: List[float], len 4
                    FTG frequencies.

            Return:
            -------
                float 
                    Reward value.

            References:
            -----------
                * 	Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
                    (p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
                    https://arxiv.org/pdf/2010.11251.pdf
        """
        # Zero command
        zero = not (self.target_dir[0] or self.target_dir[1])

        linear_vel = np.asarray(
            [self.linear_vel[0], self.linear_vel[1]], 
            dtype=np.float32
        )
        # Base horizontal linear velocity projected onto the command direction.
        proj_linear_vel = np.dot(linear_vel, self.target_dir)
        # Velocity orthogonal to the target direction.
        ort_vel = (linear_vel - proj_linear_vel * self.target_dir) \
            if zero else linear_vel
        ort_vel = np.linalg.norm(ort_vel)

        # Base horizontal angular velocity.
        h_angular_vel = np.asarray(
            [self.angular_vel[0], self.angular_vel[1]],
            dtype=np.float32
        )
        # Base angular velocity Z projected onto desired angular velocity.
        proj_angular_vel = self.angular_vel[2] * self.turn_dir

        # Set of such collision-free feet and index set of swing legs
        count_swing = 0
        foot_clear = 4
        for i in range(4):
            # If i-th foot is in swign phase.
            if ftg_freqs[i] >= self.SWIGN_PH:
                count_swing += 1

                # Verify that the height of the i-th foot is greater than the height of 
                # the surrounding terrain
                for height in self.height_scan:
                    if self.foot_target_hist[0][i][2] <= height:
                        foot_clear -= 1
                        break

        # ======================= REWARDS ======================= #
        # Linear Velocity Reward
        if zero:
            r_lv = 0
        elif proj_linear_vel < self.VEL_TH:
            r_lv = np.exp(-2 * (proj_linear_vel - self.VEL_TH) ** 2)
        else:
            r_lv = 1

        # Angular Velocity Reward
        r_av = 0
        if self.turn_dir == 0:
            r_av = 0
        elif proj_angular_vel < self.VEL_TH:
            r_av = np.exp(-1.5 * (proj_angular_vel - self.VEL_TH) ** 2)
        else:
            r_av = 1

        # Base Motion Reward
        w_2 = np.dot(h_angular_vel, h_angular_vel)
        r_b = np.exp(-1.5 * ort_vel ** 2) + np.exp(-1.5 * w_2)

        # Foot Clearance Reward
        r_fc = foot_clear / count_swing if count_swing > 0 else 1

        # Body Collision Reward
        r_bc = -sum(self.thighs_contact) - sum(self.shanks_contact)

        # Target Smoothness Reward
        r_fd_T = []
        for i in range(3):
            r_fd_T.append(np.ndarray([
                self.foot_target_hist[i][0][0], self.foot_target_hist[i][0][1], 
                self.foot_target_hist[i][0][2], self.foot_target_hist[i][1][0], 
                self.foot_target_hist[i][1][1], self.foot_target_hist[i][1][2],
                self.foot_target_hist[i][2][0], self.foot_target_hist[i][2][1], 
                self.foot_target_hist[i][2][2], self.foot_target_hist[i][3][0], 
                self.foot_target_hist[i][3][1], self.foot_target_hist[i][3][2]
            ]))
        r_s = -np.linalg.norm(r_fd_T[0] - 2.0 * r_fd_T[1] + r_fd_T[2])

        # Torque Reward
        r_tau = 0
        for pos in self.joint_angles: r_tau -= abs(pos)

        return (5*r_lv + 5*r_av + 4*r_b + r_fc + 2*r_bc + 2.5*r_s) / 100.0 + 2e-5 * r_tau

    def __update_obs(
            self, 
            time_data: timestep, 
            n_data: normal_data,
            p_data: priviliged_data
        ):
        """
            Update data from ROS
        """
        self.timestep           = time_data.timestep

        # Non-priviliged data
        self.position           = n_data.position
        self.orientation        = n_data.orientation
        self.linear_vel         = n_data.linear_vel 
        self.angular_vel        = n_data.angular_vel 
        self.toes_contact       = n_data.toes_contact 
        self.thighs_contact     = n_data.thighs_contact 
        self.shanks_contact     = n_data.shanks_contact 
        self.joint_angles       = n_data.joint_angles 
        self.joint_velocities   = n_data.joint_velocities  
        self.transform_matrices = np.reshape(n_data.transf_matrix, (4,4,4))

        # Priviliged data
        self.joint_torques   = p_data.joint_torques 
        self.normal_toe      = np.reshape(p_data.normal_toe, (4,3))
        self.toes_force1     = p_data.toes_force1     
        self.toes_force2     = p_data.toes_force2     
        self.ground_friction = p_data.ground_friction 
        self.height_scan     = np.reshape(p_data.height_scan, (4,9)) 

    def __get_obs(self): 
        return {
            # Non-priviliged Space
            'gravity_vector': self.gravity_vector,
            'angular_vel'   : self.angular_vel,
            'linear_vel'    : self.linear_vel,
            'joint_angles'  : self.joint_angles,
            'joint_vels'    : self.joint_velocities,
            'toes_contact'  : self.toes_contact,
            'thighs_contact': self.thighs_contact,
            'shanks_contact': self.shanks_contact,

            # Priviliged Space
            'normal_foot'  : self.normal_toe, 
            'height_scan'  : self.height_scan, 
            'foot_forces'  : self.toes_force1, 
            'foot_friction': self.ground_friction
        }

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        target_foot_positions, _, _ = calculate_foot_trajectories(action, self.timestep)
        joints_angles = []
        
        for i in range(4):
            r_o = target_foot_positions[i]
            T_i = self.transform_matrices[i]
            r = T_i @ np.concatenate((r_o, [1]), axis = 0)
            r = r[:3]

            leg_angles = solve_leg_IK("LEFT" if i%2 == 0 else "RIGHT", r)
            joints_angles += list(leg_angles)

        self.__actuate_joints(joints_angles)

        # TODO
        ftg_freq = [0]*4

        observation = self.__get_obs()
        reward = self.__get_reward(ftg_freq)
        done = False # TODO
        info = {}    # TODO

        return observation, reward, done, info

    def make_terrain(self, type: str, *args, **kwargs):
        # We create the terrain
        if type == 'hills': terrain = terrain_gen.hills(*args, **kwargs)
        elif type == 'steps': terrain = terrain_gen.steps(*args, **kwargs)
        elif type == 'stairs': terrain = terrain_gen.stairs(*args, **kwargs)

        # A random goal is selected
        x, y = terrain_gen.set_goal(terrain, 3)
        self.target_dir = [x / 50 - 5, y / 50 - 5]
        self.turn_dir = randint(-1, 1)

        # We store the terrain in a file
        terrain_gen.save(terrain, self.TERRAIN_FILE)

    def reset(self, terrain_file: str=''):
        """
            Reset simulation.
        """
        # Create message
        msg = text()
        msg.text = terrain_file
        self.reset_pub.publish(msg)


