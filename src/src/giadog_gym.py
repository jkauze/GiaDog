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
from src.terrain_gen import *
from src.simulation import simulation
try:
    import rospy
    import message_filters
    from spot_mini_ros.msg import joint_angles, normal_data, priviliged_data, \
        text, timestep
except:
    print(f'\033[1;93m[w]\033[0m Warning: Executing giadog_gym without ROS.')

# Controller
from src.neural_networks import *
from src.inverse_kinematics import *
from src.foot_trajectory_generator import *


# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
QUEUE_SIZE           = ENV["ROS"]["QUEUE_SIZE"]
GOAL_RADIUS_2        = ENV["TRAIN"]["GOAL_RADIUS"] ** 2
MAX_ITER_TIME        = ENV["TRAIN"]["MAX_ITERATION_TIME"]
MIN_DESIRED_VEL      = ENV["TRAIN"]["MIN_DESIRED_VEL"]
BASE_FREQ            = ENV["ROBOT"]["BASE_FREQUENCY"]
VEL_TH               = ENV["PHYSICS"]["VELOCITY_THRESHOLD"]
SWIGN_PH             = ENV["PHYSICS"]["SWING_PHASE"]
GRAVITY_VECTOR       = ENV["PHYSICS"]["GRAVITY_VECTOR"]
TERRAIN_FILE         = ENV["SIMULATION"]["TERRAIN_FILE"]
MESH_SCALE           = ENV["SIMULATION"]["MESH_SCALE"]
ROWS                 = ENV["SIMULATION"]["ROWS"]
COLS                 = ENV["SIMULATION"]["COLS"]
SIM_SECONDS_PER_STEP = ENV["SIMULATION"]["SIM_SECONDS_PER_STEP"]
HISTORY_LEN          = ENV["NEURAL_NETWORK"]["HISTORY_LEN"]


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
    FOOT_HISTORY_LEN = 3
    JOINT_VEL_HISTORY_LEN = 2
    JOINT_ERR_HISTORY_LEN = 2
    NON_PRIVILIGED_DATA = {
        'target_dir',
        'turn_dir',
        'gravity_vector',
        'angular_vel',
        'linear_vel',
        'joint_angles',
        'joint_vels',
        'ftg_phases',
        'ftg_freqs',
        'base_freq',
        'joint_err_hist',
        'joint_vel_hist',
        'foot_target_hist',
        'toes_contact',
        'thighs_contact',
        'shanks_contact'
    }
    PRIVILIGED_DATA = {
        'normal_foot',
        'height_scan',
        'foot_forces',
        'foot_friction',
        'external_force'
    }

    def __init__(self, sim: Optional[simulation]=None):
        self.observation_space = spaces.Dict({
            # Non-priviliged Space
            'target_dir': spaces.Box(
                low = -np.inf * np.ones((2,)), 
                high = np.inf * np.ones((2,)),
                dtype = np.float32
            ),
            'turn_dir': spaces.Discrete(3),
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
                low = -np.inf,
                high = np.inf,
                shape = (1,),
                dtype = np.float
            ),
            'joint_err_hist': spaces.Box(
                low = np.zeros((self.JOINT_ERR_HISTORY_LEN, 12)),
                high = 2 * np.pi * np.ones((self.JOINT_ERR_HISTORY_LEN, 12)),
                dtype = np.float
            ),
            'joint_vel_hist': spaces.Box(
                low = -np.inf * np.ones((self.JOINT_VEL_HISTORY_LEN, 12)),
                high = np.inf * np.ones((self.JOINT_VEL_HISTORY_LEN, 12)),
                dtype = np.float
            ),
            'foot_target_hist': spaces.Box(
                low = -np.inf * np.ones((self.FOOT_HISTORY_LEN, 4, 3)),
                high = np.inf * np.ones((self.FOOT_HISTORY_LEN, 4, 3)),
                dtype = np.float
            ),
            'toes_contact': spaces.MultiBinary(4),
            'thighs_contact': spaces.MultiBinary(4),
            'shanks_contact': spaces.MultiBinary(4),

            # Priviliged Space 
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
            'foot_friction': spaces.Box(
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

        self.H = np.zeros((HISTORY_LEN, controller_neural_network.NORMAL_DATA_SHAPE))
        self.foot_target_hist = np.zeros((self.FOOT_HISTORY_LEN, 4, 3))
        self.joint_vel_hist = np.zeros((self.JOINT_VEL_HISTORY_LEN, 12))
        self.joint_err_hist = np.zeros((self.JOINT_ERR_HISTORY_LEN, 12))
        self.joint_velocities = np.zeros((12,))
        self.transf_matrix = np.zeros((4,4,4))
        self.external_force  = np.zeros((3,))
        self.ftg_phases = np.zeros((8,))
        self.ftg_freqs = np.zeros((4,))
        self.base_freq = BASE_FREQ
        self.gravity_vector = GRAVITY_VECTOR
        self.target_dir = np.zeros((2,))
        self.timestep = 0
        self.E_v = []

        self.sim = sim
        if self.sim == None:
            # ROS publisher node that update the spot mini joints
            self.reset_pub = rospy.Publisher(
                'reset_simulation', 
                text, 
                queue_size=QUEUE_SIZE
            )
            # ROS publisher node that update the spot mini joints
            self.joints_pub = rospy.Publisher(
                'spot_joints', 
                joint_angles, 
                queue_size=QUEUE_SIZE
            )
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
                queue_size=QUEUE_SIZE,
                slop=0.1,
                allow_headerless=True
            )
            ts.registerCallback(self.__update_obs_ros)
        else:
            self.count = 0
            self.begin_time = time()
            self.__update_obs_sim()

    def __actuate_joints(self, joint_target_positions: List[float]):
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
        if self.sim == None:
            # Create message
            msg = joint_angles()
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
        zero = not (self.target_dir[0] or self.target_dir[1])

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
        r_fd_T = np.reshape(self.foot_target_hist, (self.FOOT_HISTORY_LEN,-1))
        r_s = -np.linalg.norm(r_fd_T[0] - 2.0 * r_fd_T[1] + r_fd_T[2])

        # Torque Reward
        r_tau = -sum(abs(torque) for torque in state.joint_torques)

        reward = (5*r_lv + 5*r_av + 4*r_b + r_fc + 2*r_bc + 2.5*r_s) \
            / 100.0 + 2e-5 * r_tau

        return reward

    def __update_obs_ros(self, time_data, n_data, p_data):
        """
            Update data from ROS
        """
        self.timestep    = time_data.timestep
        self.position    = n_data.position
        self.orientation = n_data.orientation
        self.command_dir = self.target_dir - np.array(self.position[:2])
        self.command_dir = self.command_dir / np.linalg.norm(self.command_dir)

        # Non-priviliged data
        self.linear_vel            = n_data.linear_vel 
        self.angular_vel           = n_data.angular_vel 
        self.toes_contact          = n_data.toes_contact 
        self.thighs_contact        = n_data.thighs_contact 
        self.shanks_contact        = n_data.shanks_contact 
        self.joint_angles          = n_data.joint_angles 
        N = self.JOINT_VEL_HISTORY_LEN
        self.joint_vel_hist[1:N]   = self.joint_vel_hist[0:N-1]
        self.joint_vel_hist[0]     = self.joint_velocities
        self.joint_velocities      = n_data.joint_velocities  
        self.transf_matrix    = np.reshape(n_data.transf_matrix, (4,4,4))
        N = self.FOOT_HISTORY_LEN
        self.foot_target_hist[1:N] = self.foot_target_hist[0:N-1]
        self.foot_target_hist[0]   = np.reshape(n_data.foot_target, (4,3))

        # Priviliged data
        self.joint_torques   = p_data.joint_torques 
        self.normal_toe      = np.reshape(p_data.normal_toe, (4,3))
        self.toes_force1     = p_data.toes_force1     
        self.toes_force2     = p_data.toes_force2     
        self.ground_friction = p_data.ground_friction 
        self.height_scan     = np.reshape(p_data.height_scan, (4,9)) 
        self.external_force  = np.zeros((3,))

        v = int(np.array(self.linear_vel[:2]) @ self.command_dir > MIN_DESIRED_VEL)
        self.E_v.append(v)

    def __update_obs_sim(self):
        """
            Update data from simulation
        """
        self.sim.p.stepSimulation()
        self.sim.update_sensor_output()

        self.timestep += SIM_SECONDS_PER_STEP

        self.position    = self.sim.position
        self.orientation = self.sim.orientation
        self.command_dir = self.target_dir - np.array(self.position[:2])
        self.command_dir = self.command_dir / np.linalg.norm(self.command_dir)

        N = self.JOINT_VEL_HISTORY_LEN
        self.joint_vel_hist[1:N]   = self.joint_vel_hist[0:N-1]
        self.joint_vel_hist[0]     = self.joint_velocities
        self.joint_velocities      = self.sim.joint_velocities 

        N = self.FOOT_HISTORY_LEN
        self.foot_target_hist[1:N] = self.foot_target_hist[0:N-1]
        self.foot_target_hist[0]   = np.reshape(self.sim.foot_target, (4,3))

        self.transf_matrix = self.sim.transf_matrix

        if self.count == 1000:
            vel = 1000 / (time() - self.begin_time)
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

        return d < GOAL_RADIUS_2 or self.timestep > MAX_ITER_TIME\
                 or self.sim.is_fallen()

    def get_obs(self) -> Dict[str, Any]: 
        """
            [TODO]
        """
        state = self if self.sim == None else self.sim

        return {
            # Non-priviliged Space
            'target_dir'       : self.target_dir,
            'turn_dir'         : self.turn_dir,
            'gravity_vector'   : self.gravity_vector,
            'angular_vel'      : state.angular_vel,
            'linear_vel'       : state.linear_vel,
            'joint_angles'     : state.joint_angles,
            'joint_vels'       : state.joint_velocities,
            'ftg_phases'       : self.ftg_phases,
            'ftg_freqs'        : self.ftg_freqs,
            'base_freq'        : self.base_freq,
            'joint_err_hist'   : self.joint_err_hist,
            'joint_vel_hist'   : self.joint_vel_hist,
            'foot_target_hist' : self.foot_target_hist,
            'toes_contact'     : state.toes_contact,
            'thighs_contact'   : state.thighs_contact,
            'shanks_contact'   : state.shanks_contact,

            # Priviliged Space
            'normal_foot'    : state.normal_toe, 
            'height_scan'    : state.height_scan, 
            'foot_forces'    : state.toes_force1, 
            'foot_friction'  : state.ground_friction,
            'external_force' : self.external_force
        }

    def step(self, action: np.ndarray) -> Tuple[dict, float, bool, dict]:
        """
            Apply an action on the environment

            [TODO]
        """
        ftg_data = calculate_foot_trajectories(action, self.timestep)
        target_foot_positions, self.ftg_freqs, self.ftg_phases = ftg_data
        self.ftg_phases = np.reshape(self.ftg_phases, -1)

        joints_angles = []
        for i in range(4):
            r_o = target_foot_positions[i]
            T_i = self.transf_matrix[i]
            r = T_i @ np.concatenate((r_o, [1]), axis = 0)
            r = r[:3]

            leg_angles = solve_leg_IK("LEFT" if i%2 == 0 else "RIGHT", r)
            joints_angles += list(leg_angles)

        self.__actuate_joints(joints_angles)

        observation = self.get_obs()
        reward = self.__get_reward()
        done = self.__terminate()
        info = {}    # TODO

        if self.sim != None: self.__update_obs_sim()

        return observation, reward, done, info

    def make_terrain(self, type: str, *args, **kwargs):
        """
            [TODO]
        """
        # We create the terrain
        if type == 'hills': terrain = terrain_gen.hills(*args, **kwargs)
        elif type == 'steps': terrain = terrain_gen.steps(*args, **kwargs)
        elif type == 'stairs': terrain = terrain_gen.stairs(*args, **kwargs)

        # A random goal is selected
        x, y = terrain_gen.set_goal(terrain, 3)
        x = x / MESH_SCALE[0] - ROWS / (2 * MESH_SCALE[0])
        y = y / MESH_SCALE[1] - COLS / (2 * MESH_SCALE[1])
        self.target_dir = np.array([x, y])
        self.turn_dir = randint(-1, 1)

        # We store the terrain in a file
        terrain_gen.save(terrain, TERRAIN_FILE)

    def reset(self, terrain_file: str=''):
        """
            Reset simulation.
        """
        if self.sim == None:
            # Create message
            msg = text()
            msg.text = terrain_file
            self.reset_pub.publish(msg)
        else:
            self.sim.reset(terrain_file)
            self.timestep = 0
            self.count = 0
            self.begin_time = time()

        self.E_v = []

    def traverability(self) -> float:
        """
            Calculate the current transversability
        """
        if len(self.E_v) == 0: return 0
        return sum(self.E_v) / len(self.E_v)
