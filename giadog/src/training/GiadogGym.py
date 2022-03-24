"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    Description Gym enviroment for the training of the control policies
"""

# Utilities
import numpy as np
from typing import *
from time import time
from random import randint

# OpenIA Gym
import gym
from gym import spaces

# Simulation
from src.simulation import Simulation
try:
    import rospy
    import message_filters
    from spot_mini_ros.msg import JointAngles, NonPriviliged, ExtraData, \
        Priviliged, Text, Timestep
except:
    print(f'\033[1;93m[w]\033[0m Warning: Executing giadog_gym without ROS.')

# Controller
from src.kinematics import *
from src.simulation import *
from src.__env__ import QUEUE_SIZE, GOAL_RADIUS_2, MAX_ITER_TIME, \
    MIN_DESIRED_VEL, BASE_FREQ, VEL_TH, SWIGN_PH, GRAVITY_VECTOR, \
    TERRAIN_FILE, MESH_SCALE, ROWS, COLS, HISTORY_LEN, FOOT_HISTORY_LEN, \
    JOINT_VEL_HISTORY_LEN, JOINT_ERR_HISTORY_LEN


class TeacherEnv(gym.Env):
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

    def __init__(self, sim: Optional[Simulation]=None):
        """
            [TODO]
        """
        # We set the observation and action space
        self.observation_space = spaces.Dict({
            # Non-priviliged Space
            'command_dir': spaces.Box(
                low = -np.inf * np.ones((2,)), 
                high = np.inf * np.ones((2,)),
                dtype = np.float32
            ),
            'turn_dir': spaces.Box(
                low = -np.ones((1,)), 
                high = np.ones((1,)),
                dtype = np.int8
            ),
            'gravity_vector': spaces.Box(
                low = -np.inf * np.ones((3,)), 
                high = np.inf * np.ones((3,)),
                dtype = np.float32
            ),
            'angular_vel': spaces.Box(
                low = -np.inf * np.ones((3,)), 
                high = np.inf * np.ones((3,)),
                dtype = np.float32
            ),
            'linear_vel': spaces.Box(
                low = -np.inf * np.ones((3,)), 
                high = np.inf * np.ones((3,)),
                dtype = np.float32
            ),
            'joint_angles': spaces.Box(
                low = np.zeros((12,)), 
                high = 2 * np.pi * np.ones((12,)),
                dtype = np.float32
            ),
            'joint_vels': spaces.Box(
                low = -np.inf * np.ones((12,)), 
                high = np.inf * np.ones((12,)),
                dtype = np.float32
            ),
            'ftg_phases': spaces.Box(
                low = -np.ones((8,)),
                high = np.ones((8,)),
                dtype = np.float32
            ),
            'ftg_freqs': spaces.Box(
                low = -np.inf * np.ones((4,)), 
                high = np.inf * np.ones((4,)),
                dtype = np.float32
            ),
            'base_freq': spaces.Box(
                low = -np.inf * np.ones((1,)),
                high = np.inf * np.ones((1,)),
                shape = (1,),
                dtype = np.float
            ),
            'joint_err_hist': spaces.Box(
                low = np.zeros((JOINT_ERR_HISTORY_LEN, 12)),
                high = 2 * np.pi * np.ones((JOINT_ERR_HISTORY_LEN, 12)),
                dtype = np.float
            ),
            'joint_vel_hist': spaces.Box(
                low = -np.inf * np.ones((JOINT_VEL_HISTORY_LEN, 12)),
                high = np.inf * np.ones((JOINT_VEL_HISTORY_LEN, 12)),
                dtype = np.float
            ),
            'feet_target_hist': spaces.Box(
                low = -np.inf * np.ones((FOOT_HISTORY_LEN, 4, 3)),
                high = np.inf * np.ones((FOOT_HISTORY_LEN, 4, 3)),
                dtype = np.float
            ),
            'toes_contact': spaces.Box(
                low = np.zeros((4,)), 
                high = np.ones((4,)),
                dtype = np.int8
            ),
            'thighs_contact': spaces.Box(
                low = np.zeros((4,)), 
                high = np.ones((4,)),
                dtype = np.int8
            ),
            'shanks_contact': spaces.Box(
                low = np.zeros((4,)), 
                high = np.ones((4,)),
                dtype = np.int8
            ),

            # Privileged Space 
            'normal_foot': spaces.Box(
                low = -np.inf * np.ones((4, 3)), 
                high = np.inf * np.ones((4, 3)),
                dtype = np.float32
            ), 
            'height_scan': spaces.Box(
                low = -np.inf * np.ones((4, 9)), 
                high = np.inf * np.ones((4, 9)),
                dtype = np.float32
            ), 
            'foot_forces': spaces.Box(
                low = -np.inf * np.ones((4,)), 
                high = np.inf * np.ones((4,)),
                dtype = np.float32
            ), 
            'foot_friction': spaces.Box(
                low = -np.inf * np.ones((4,)), 
                high = np.inf * np.ones((4,)),
                dtype = np.float32
            ),
            'external_force': spaces.Box(
                low = -np.inf * np.ones((3,)), 
                high = np.inf * np.ones((3,)),
                dtype = np.float32
            )
        })
        self.action_space = spaces.Box(
            low = -float('inf') * np.ones((16,)),
            high = float('inf') * np.ones((16,)),
            dtype = np.float32
        )

        self.__reset_state()

        # We check the representation of the environment
        self.sim = sim
        if self.sim == None:
            # ROS publisher node that update the spot mini joints
            self.reset_pub = rospy.Publisher(
                'reset_simulation', 
                Text, 
                queue_size=QUEUE_SIZE
            )
            # ROS publisher node that update the spot mini joints
            self.joints_pub = rospy.Publisher(
                'spot_joints', 
                JointAngles, 
                queue_size=QUEUE_SIZE
            )
            non_priviliged_data_sub = message_filters.Subscriber(
                'nono_priviliged_data', 
                NonPriviliged
            )
            priviliged_data_sub = message_filters.Subscriber(
                'priviliged_data', 
                Priviliged
            )
            extra_data_sub = message_filters.Subscriber(
                'extra_data', 
                ExtraData
            )
            timestep_sub = message_filters.Subscriber(
                'timestep',
                Timestep
            )
            ts = message_filters.ApproximateTimeSynchronizer(
                [
                    timestep_sub, 
                    non_priviliged_data_sub, 
                    priviliged_data_sub,
                    extra_data_sub
                ], 
                queue_size=QUEUE_SIZE,
                slop=0.1,
                allow_headerless=True
            )
            ts.registerCallback(self.__update_obs_ros)
        else:
            self.count = 0
            self.reset(TERRAIN_FILE) 
            self.begin_time = time()
            self.__update_obs_sim()

    def __reset_state(self):
        """ 
            [TODO] 
        """
        # Non-priviliged data
        self.command_dir      : np.array = np.ones((2,))
        self.turn_dir         : np.array = np.zeros((1,))
        self.gravity_vector   : np.array = np.array(GRAVITY_VECTOR)
        self.linear_vel       : np.array = np.zeros((3,))
        self.angular_vel      : np.array = np.zeros((3,))
        self.joint_angles     : np.array = np.zeros((12,))
        self.joint_velocities : np.array = np.zeros((12,))
        self.ftg_phases       : np.array = np.zeros((8,))
        self.ftg_freqs        : np.array = np.zeros((4,))
        self.base_freq        : np.array = np.array([BASE_FREQ])
        self.joint_err_hist   : np.array = np.zeros((JOINT_ERR_HISTORY_LEN, 12))
        self.joint_vel_hist   : np.array = np.zeros((JOINT_VEL_HISTORY_LEN, 12))
        self.feet_target_hist : np.array = np.zeros((FOOT_HISTORY_LEN, 4, 3))
        self.toes_contact     : np.array = np.zeros((4,))
        self.thighs_contact   : np.array = np.zeros((4,))
        self.shanks_contact   : np.array = np.zeros((4,))

        # Priviliged data
        self.normal_toe      : np.array = np.zeros((4,3))
        self.height_scan     : np.array = np.zeros((4,9))
        self.toes_force1     : np.array = np.zeros((4,))   
        self.toes_force2     : np.array = np.zeros((4,))
        self.ground_friction : np.array = np.zeros((4,))
        self.external_force  : np.array = np.zeros((3,))

        # Other data
        self.timestep         : float       = 0.0
        self.is_fallen        : bool        = False
        self.position         : List[float] = [0.0, 0.0, 0.0]
        self.orientation      : List[float] = [0.0, 0.0, 0.0]
        self.trajectory       : List[float] = []
        self.H                : np.array    = np.zeros((HISTORY_LEN, 60))
        self.transf_matrices  : np.array    = np.zeros((4,4,4)) 
        self.joint_torques    : np.array    = np.zeros((12,))
        self.target_dir       : np.array    = np.zeros((2,))

    def __actuate_joints(self, joint_target_positions: List[float]):
        """
            Moves the robot joints to a given target position.

            Argument:
            ---------
                joint_target_positions: List[float], shape (12,)
                    Quadruped joints desired angles. 
        """
        if self.sim == None:
            # Create message
            msg = JointAngles()
            msg.joint_target_positions = joint_target_positions
            self.joints_pub.publish(msg)
        else:
            self.sim.actuate_joints(joint_target_positions)

    def __get_reward(self) -> float:
        """
            Reward function.

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
        state = self if self.sim == None else self.sim

        # Zero command
        zero = not (self.command_dir[0] or self.command_dir[1])

        linear_vel = state.linear_vel[:2]
        # Base horizontal linear velocity projected onto the command direction.
        proj_linear_vel = np.dot(linear_vel, self.command_dir)
        # Velocity orthogonal to the target direction.
        ort_vel = (linear_vel - proj_linear_vel * self.command_dir) \
            if zero else linear_vel
        ort_vel = np.linalg.norm(ort_vel)

        # Base horizontal angular velocity.
        h_angular_vel = state.angular_vel[:2]
        # Base angular velocity Z projected onto desired angular velocity.
        proj_angular_vel = state.angular_vel[2] * self.turn_dir

        # Set of such collision-free feet and index set of swing legs
        count_swing = 0
        foot_clear = 0
        for i in range(4):
            # If i-th foot is in swign phase.
            if self.ftg_freqs[i] >= SWIGN_PH:
                count_swing += 1

                # Verify that the height of the i-th foot is greater than the height of 
                # the surrounding terrain
                foot_clear += all(height < 0 for height in state.height_scan[i])

        # ======================= REWARDS ======================= #
        # Linear Velocity Reward
        if zero:
            r_lv = 0
        elif proj_linear_vel < VEL_TH:
            r_lv = np.exp(-2 * (proj_linear_vel - VEL_TH) ** 2)
        else:
            r_lv = 1

        # Angular Velocity Reward
        if proj_angular_vel < VEL_TH:
            r_av = np.exp(-1.5 * (proj_angular_vel - VEL_TH) ** 2)
        else:
            r_av = 1

        # Base Motion Reward
        w_2 = np.dot(h_angular_vel, h_angular_vel)
        r_b = np.exp(-1.5 * ort_vel ** 2) + np.exp(-1.5 * w_2)

        # Foot Clearance Reward
        r_fc = foot_clear / count_swing if count_swing > 0 else 1

        # Body Collision Reward
        r_bc = - sum(state.thighs_contact) - sum(state.shanks_contact)

        # Target Smoothness Reward
        r_fd_T = np.reshape(self.feet_target_hist, (FOOT_HISTORY_LEN,-1))
        r_s = -np.linalg.norm(r_fd_T[0] - 2.0 * r_fd_T[1] + r_fd_T[2])

        # Torque Reward
        r_tau = -sum(abs(torque) for torque in state.joint_torques)

        reward = (5*r_lv + 5*r_av + 4*r_b + r_fc + 2*r_bc + 2.5*r_s) \
            / 100.0 + 2e-5 * r_tau

        return reward

    def __update_obs_ros(self, time_data, n_data, p_data, o_data):
        """
            Update data from ROS
        """
        # Other data
        self.position    = o_data.position
        self.orientation = o_data.orientation

        # Non-priviliged data
        self.linear_vel       = n_data.linear_vel 
        self.angular_vel      = n_data.angular_vel 
        self.joint_angles     = n_data.joint_angles 

        N = JOINT_VEL_HISTORY_LEN
        self.joint_vel_hist[1:N] = self.joint_vel_hist[0:N-1]
        self.joint_vel_hist[0]   = self.joint_velocities

        self.joint_velocities = n_data.joint_velocities 
        self.toes_contact     = n_data.toes_contact 
        self.thighs_contact   = n_data.thighs_contact 
        self.shanks_contact   = n_data.shanks_contact 

        self.command_dir = self.target_dir - np.array(self.position[:2])
        self.command_dir = self.command_dir / np.linalg.norm(self.command_dir)

        # Priviliged data
        self.normal_toe      = np.reshape(p_data.normal_toe, (4,3))
        self.toes_force1     = p_data.toes_force1    
        self.toes_force2     = p_data.toes_force2
        self.ground_friction = p_data.ground_friction
        self.height_scan     = np.reshape(p_data.height_scan, (4,9))
        self.external_force  = p_data.external_force

        # Other data
        self.timestep         = time_data.timestep
        self.is_fallen        = o_data.is_fallen
        self.transf_matrices  = np.reshape(o_data.transf_matrices, (4,4,4))
        self.joint_torques    = o_data.joint_torques
        v = int(np.array(self.linear_vel[:2]) @ self.command_dir > MIN_DESIRED_VEL)
        self.trajectory.append(v)

    def __update_obs_sim(self):
        """
            Update data from simulation
        """
        self.sim.step()
        self.sim.update_sensor_output()

        # Other data
        self.position    = self.sim.position
        self.orientation = self.sim.orientation

        # Non-priviliged data
        self.linear_vel       = self.sim.linear_vel 
        self.angular_vel      = self.sim.angular_vel 
        self.joint_angles     = self.sim.joint_angles 

        N = JOINT_VEL_HISTORY_LEN
        self.joint_vel_hist[1:N] = self.joint_vel_hist[0:N-1]
        self.joint_vel_hist[0]   = self.joint_velocities

        self.joint_velocities = self.sim.joint_velocities 
        self.toes_contact     = self.sim.toes_contact 
        self.thighs_contact   = self.sim.thighs_contact 
        self.shanks_contact   = self.sim.shanks_contact 

        self.command_dir = self.target_dir - self.position[:2]
        self.command_dir = self.command_dir / np.linalg.norm(self.command_dir)

        # Priviliged data
        self.normal_toe      = self.sim.normal_toe
        self.toes_force1     = self.sim.toes_force1    
        self.toes_force2     = self.sim.toes_force2
        self.ground_friction = self.sim.ground_friction
        self.height_scan     = self.sim.height_scan
        self.external_force  = self.sim.external_force

        # Other data
        self.timestep         = self.sim.timestep
        self.is_fallen        = self.sim.is_fallen
        self.transf_matrices  = self.sim.transf_matrices
        self.joint_torques    = self.sim.joint_torques
        v = int(self.linear_vel[:2] @ self.command_dir > MIN_DESIRED_VEL)
        self.trajectory.append(v)

        if self.count == 100:
            vel = 100 / (time() - self.begin_time)
            print(f'Executing approximately {vel} steps per second.')
            self.count = 0
            self.begin_time = time()
        else:
            self.count += 1

    def __terminate(self) -> bool:
        """
            Check if the iteration finished
        """
        # We calculate the distance between the position of the robot and the target
        d_vector = self.target_dir - np.array(self.position[:2])
        d = d_vector @ d_vector

        return d < GOAL_RADIUS_2 or self.timestep > MAX_ITER_TIME or self.is_fallen

    def get_obs(self) -> Dict[str, Any]: 
        """
            [TODO]
        """
        return {
            # Non-priviliged Data
            'command_dir'      : self.command_dir,
            'turn_dir'         : self.turn_dir,
            'gravity_vector'   : self.gravity_vector,
            'angular_vel'      : self.angular_vel,
            'linear_vel'       : self.linear_vel,
            'joint_angles'     : self.joint_angles,
            'joint_vels'       : self.joint_velocities,
            'ftg_phases'       : self.ftg_phases,
            'ftg_freqs'        : self.ftg_freqs,
            'base_freq'        : self.base_freq,
            'joint_err_hist'   : self.joint_err_hist,
            'joint_vel_hist'   : self.joint_vel_hist,
            'feet_target_hist' : self.feet_target_hist,
            'toes_contact'     : self.toes_contact,
            'thighs_contact'   : self.thighs_contact,
            'shanks_contact'   : self.shanks_contact,

            # Priviliged Data
            'normal_foot'    : self.normal_toe, 
            'height_scan'    : self.height_scan, 
            'foot_forces'    : self.toes_force1, 
            'foot_friction'  : self.ground_friction,
            'external_force' : self.external_force
        }

    def step(self, action: np.array) -> Tuple[dict, float, bool, dict]:
        """
            Apply an action on the environment

            [TODO]
        """
        ftg_data = foot_trajectories(action, self.timestep)
        foot_target_pos, self.ftg_freqs, self.ftg_phases = ftg_data
        self.ftg_phases = np.reshape(self.ftg_phases, -1)

        joints_angles_target = []
        for i in range(4):
            r_o = foot_target_pos[i]
            T_i = self.transf_matrices[i]
            r = T_i @ np.concatenate((r_o, [1]), axis = 0)
            r = r[:3]

            leg_angles = solve_leg_IK("LEFT" if i%2 == 0 else "RIGHT", r)
            joints_angles_target += list(leg_angles)

        self.__actuate_joints(joints_angles_target)

        observation = self.get_obs()
        reward = self.__get_reward()
        done = self.__terminate()
        info = {}    # TODO

        if self.sim != None: self.__update_obs_sim()

        N = FOOT_HISTORY_LEN
        self.feet_target_hist[1:N] = self.feet_target_hist[0:N-1]
        self.feet_target_hist[0]   = np.array(foot_target_pos)

        N = JOINT_ERR_HISTORY_LEN
        self.joint_err_hist[1:N] = self.joint_err_hist[0:N-1]
        joint_err = np.abs(np.array(joints_angles_target) - np.array(self.joint_angles))
        self.joint_err_hist[0]   = np.reshape(joint_err, (4,3))

        return observation, reward, done, info

    def make_terrain(self, type: str, *args, **kwargs):
        """
            [TODO]
        """
        # We create the terrain
        if type == 'hills': terrain = hills(*args, **kwargs)
        elif type == 'steps': terrain = steps(*args, **kwargs)
        elif type == 'stairs': terrain = stairs(*args, **kwargs)

        # A random goal is selected
        x, y = set_goal(terrain, 3)
        x = x / MESH_SCALE[0] - ROWS / (2 * MESH_SCALE[0])
        y = y / MESH_SCALE[1] - COLS / (2 * MESH_SCALE[1])
        self.target_dir = np.array([x, y])
        self.turn_dir = randint(-1, 1)

        # We store the terrain in a file
        save_terrain(terrain, TERRAIN_FILE)

    def reset(self, terrain_file: str=''):
        """
            Reset simulation.
        """
        self.__reset_state()

        if self.sim == None:
            # Create message
            msg = Text()
            msg.text = terrain_file
            self.reset_pub.publish(msg)
            sleep(5)
            self.__update_obs_ros()
        else:
            self.sim.reset(terrain_file)
            self.timestep = 0
            self.count = 0
            self.begin_time = time()
            self.__update_obs_sim()

        self.trajectory = []
        return self.get_obs()


    def traverability(self) -> float:
        """
            Calculate the current transversability
        """
        if len(self.trajectory) == 0 or self.is_fallen: return 0
        return sum(self.trajectory) / len(self.trajectory)
