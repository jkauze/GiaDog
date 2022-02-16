"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file provides an interface to control and monitor the simulation status.
"""

# Utilities
import json
import time
from typing import *

# Simulacion
import pybullet as p
from src.bullet_dataclasses import *

# Array usage
import numpy as np

# Inverse kinematics testing
from src.inverse_kinematics import solve_leg_IK
from src.foot_trajectory_generator import calculate_foot_trajectories
from src.transformation_matrix_calculator import \
    get_leg_to_horizontal_frame_transformations


# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
GRAVITY_VECTOR = np.array(ENV["PHYSICS"]["GRAVITY_VECTOR"])
MESH_SCALE     = ENV["SIMULATION"]["MESH_SCALE"]
JOINTS_IDS     = ENV["SIMULATION"]["JOINTS_IDS"]
HIPS_IDS       = ENV["SIMULATION"]["HIPS_IDS"]
THIGHS_IDS     = ENV["SIMULATION"]["THIGHS_IDS"]
SHANKS_IDS     = ENV["SIMULATION"]["SHANKS_IDS"]
TOES_IDS       = ENV["SIMULATION"]["TOES_IDS"]  


class simulation:
    """ Control and monitor the simulation of the spot-mini in pybullet. """
    def __init__(
            self,
            giadog_urdf_file: str,
            bullet_server,
            gui: bool=False,
            self_collision_enabled: bool=False,
        ): 
        """
            Arguments:
            ----------
                giadog_urdf_file: str 
                    Path to the URDF file of the quadruped robot.

                bullet_server: module 
                    Pybullet module.

                gui: bool, optional
                    Indicates if the simulation GUI will be displayed.
                    Default: False

                self_collision_enabled: bool, optional
                    TODO
                    Default: False
        """
        self.giadog_urdf_file = giadog_urdf_file
        self.p = bullet_server
        self.self_collision_enabled = self_collision_enabled
        
        self.p.connect(self.p.GUI if gui else self.p.DIRECT, options="--opengl2")

    def __get_terrain_height(self, x: float, y: float) -> float:
        """
            Returns the height of the terrain at x,y coordinates (in cartesian world 
            coordiantes). This function assumes the terrain has no bridge 'like' 
            structures.

            Arguments:
            ----------
                x: float 
                    x position in cartesian global coordinates.

                y: float
                    y position in cartesian global coordinates.
            
            Return:
            -------
                float 
                    The terrain height at that x, y point.
                    numpy.NaN if if the coordinates go off the map
        """
        x = int(x / MESH_SCALE[0]) + self.center[0]
        y = int(y / MESH_SCALE[1]) + self.center[1]

        rows, cols = self.terrain_array.shape
        if x < 0 or x >= rows or y < 0 or y >= cols: return np.NaN

        return self.terrain_array[x][y] - self.z_diff 

    def __reset_state(self):
        """
            Reset bot state.
        """
        self.position                  = np.zeros([3])
        self.orientation               = np.zeros([2])

        # State data // Sensor data
        self.desired_direction         = np.zeros([2])
        self.desired_turning_direction = np.zeros([1])
        self.base_linear_velocity      = np.zeros([3])
        self.base_angular_velocity     = np.zeros([3])
        self.base_rpy                  = np.zeros([3])
        self.joint_angles              = np.zeros([12]) 
        self.joint_velocities          = np.zeros([12])

        # FTG (These may be provided by the C++ controller module)
        self.ftg_phases_sin_cos = np.zeros([4,2])
        self.ftg_frequencies    = np.zeros([4])
        self.base_frequency     = np.zeros([1])
        
        # Priviledge data
        self.terrain_normal_at_each_toe = np.zeros([4, 3])
        self.normal_force_at_each_toe   = np.zeros([4]) # (Foot contact forces?¿)
        self.toes_contact_states        = np.zeros([4], dtype=np.int8)
        self.thighs_contact_states      = np.zeros([4], dtype=np.int8)
        self.shanks_contact_states      = np.zeros([4], dtype=np.int8)
        self.height_scan_at_each_toe    = np.zeros([4, 9])
        self.contact_force_at_each_toe  = np.zeros([4])
        self.foot_ground_friction_coefficients = np.zeros([4])

        # For debug:
        self.height_scan_lines = np.zeros([4,9,2,3])
        self.external_force_applied_to_the_base = np.zeros([3])

        # Data only for reward purposes
        self.joint_torques = np.zeros(12)

        # Extra sensor (This may be used in the future)
        self.toe_force_sensor = np.zeros(4) 

        # Transformation matrices from the hip to the leg base
        self.transformation_matrices = np.zeros((4,4,4))

        self.foot_target = np.zeros((4,3))

    @staticmethod
    def __get_foot_height_scan_coordinates(x: float, y: float, alpha: float) -> np.array:
        """
            Given a robot toe position and orientation, returns the positions of the toe 
            height sensor corrdinates.

            Arguments:
            ----------
                x: float
                    x coordinate of the robot toe. [In the world frame]
                y: float
                    y coordinate of the robot toe. [In the world frame]
                alpha: float
                    Orientation of the toe.

            Return:
            -------
                numpy.array, shape (9, 2)
                    Array with each of the toe height sensor coordinates.
        """
        n = 9      # Number of points around each foot
        r = 0.07   # Radius of the height sensors around each toe.
        P = np.empty([n, 2])
        phi = 2*np.pi/n

        for i in range(n):
            angle_i = alpha + i* phi
            P[i] = np.array([x + r * np.cos(angle_i), y +  r * np.sin(angle_i)])
        
        return P

    @staticmethod
    def __contact_info_average(
            contact_points_info: List[ContactInfo]
        ) -> Tuple[float, float, np.array]: 
        """
            Given a robot toe position and orientation, returns the positions of the toe 
            height sensor coordinates.

            Arguments:
            ----------
                contact_points_info: List[ContactInfo] 
                    List containing the contact info of each point that has contact with 
                    the leg foot.
            
            Returns:
            --------
                float
                    magnitude of the normmal force on the foot.

                float
                    Friction coeficient between the foot and the terrain.

                numpy.array, shape (3,)
                    direction of the normal force accting on the foot.
        """
        contact_force  = np.array([0,0,0]) 
        friction_force = np.array([0,0,0]) 

        for contact_info in contact_points_info:
            contact_force = contact_force + contact_info.normalForce *\
                 np.array(contact_info.contactNormalOnB) 
            friction_1 = contact_info.lateralFriction1 * \
                np.array(contact_info.lateralFrictionDir1)
            friction_2 = contact_info.lateralFriction2 * \
                np.array(contact_info.lateralFrictionDir2)

            friction_force = friction_force + friction_1 + friction_2

        contact_force_mag = np.sqrt(contact_force.dot(contact_force))
        fricction_coefficient = np.sqrt(friction_force.dot(friction_force))
        fricction_coefficient /= contact_force_mag
        contact_force /= contact_force_mag

        return (contact_force_mag, fricction_coefficient, contact_force)

    def __initialize(self, x_o: float=0.0, y_o: float=0.0, fix_robot_base: bool=False):
        """
            Initializes a pybullet simulation, setting up the terrain, gravity and the 
            quadruped in the pybullet enviroment. (And enabiling the torque sensors in 
            the quadruped foots/toes)

            Arguments:
            ----------
                x_o: float, optional
                    x coordintate of the robot initial position (In the world frame).
                    Default: 0.0

                y_o: float, optional
                    y coordintate of the robot initial position (In the world frame).
                    Default: 0.0

                fix_robot_base: bool, optional
                    [TODO]
                    Default: False
        """
        # Create terrain object
        terrain_shape = self.p.createCollisionShape(
            shapeType = self.p.GEOM_HEIGHTFIELD, 
            meshScale = MESH_SCALE,
            fileName  = self.terrain_file, 
            heightfieldTextureScaling=128
        )
        self.terrain = p.createMultiBody(0, terrain_shape)
        self.p.resetBasePositionAndOrientation(self.terrain, [0,0,0], [0,0,0,1])
        self.p.setGravity(*GRAVITY_VECTOR)

        # Get difference between terrain array and real terrain
        ray_info = self.p.rayTest((0, 0, -50),(0, 0, 50))[0]
        self.z_diff = self.terrain_array[self.center[0]][self.center[1]] - ray_info[3][-1]

        # Obtain the maximum height around the starting point
        z_o = -50.0
        x = x_o - 0.2
        while x <= x_o + 0.2:
            y = y_o - 0.2
            while y <= y_o + 0.2:
                z_o = max(z_o, self.__get_terrain_height(x, y))
                y += 0.05
            x += 0.05

        # Load mini-spot from URDF file.
        print(f'\033[1;36m[i]\033[0m Initial position: ({x_o}, {y_o}, {z_o})')
        if self.self_collision_enabled:
            self.quadruped = self.p.loadURDF(
                self.giadog_urdf_file, 
                [x_o, y_o, self.__get_terrain_height(x_o, y_o) + 0.3],
                flags = self.p.URDF_USE_SELF_COLLISION | \
                   self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
                   useFixedBase = fix_robot_base,
            )
        else:
            self.quadruped = self.p.loadURDF(
                self.giadog_urdf_file, 
                [x_o, y_o, self.__get_terrain_height(x_o, y_o) + 0.3],
                useFixedBase = fix_robot_base,
            )

        # Torque sensors are enable on the quadruped toes
        for toe_id in TOES_IDS:
            self.p.enableJointForceTorqueSensor(
                bodyUniqueId = self.quadruped,
                jointIndex   = toe_id,
                enableSensor = True,
            )

    def reset(self, terrain_file: str, x_o: float=0.0, y_o: float=0.0):
        """
            Reset simulation.

            Arguments:
            ----------
                terrain_file: str, optional
                    Path to the .txt file representing the terrain.

                x_o: float, optional
                    x coordintate of the robot initial position (In the world frame).
                    Default: 0.0
                    
                y_o: float, optional
                    y coordintate of the robot initial position (In the world frame).
                    Default: 0.0
        """
        self.terrain_file = terrain_file
        
        # This array is used to calculate the robot toes heightfields 
        # Note : The last column is ignored because numpy adds a column of nans while 
        # reading the file
        self.terrain_array = np.genfromtxt(self.terrain_file,  delimiter=",")[:, :-1]
        center_x, center_y = self.terrain_array.shape
        self.center = (center_x // 2, center_y // 2)

        self.p.resetSimulation()

        self.__reset_state()
        self.__initialize(x_o, y_o)

    def actuate_joints(self, joint_target_positions: List[float]):
        """
            Moves the robot joints to a given target position.

            Arguments:
            ---------
                joint_target_positions: List[float], shape (12,)
                    Quadruped joints desired angles. 
                    The order is the same as for the robot joints_ids.
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

                    Note: It may be useful to add the Kp and Kd as inputs
        """
        try:
            self.p.setJointMotorControlArray(
                bodyUniqueId = self.quadruped,
                jointIndices = JOINTS_IDS,
                controlMode  = self.p.POSITION_CONTROL,
                targetPositions = joint_target_positions,
            )
        except Exception as e:
            print(f'\033[1;93m[w]\033[0m {e}.')


    # =========================== UPDATE FUNCTIONS =========================== #
    def update_position_orientation(self):
        self.position, self.orientation = \
            self.p.getBasePositionAndOrientation(self.quadruped)
        self.orientation = np.array(self.p.getEulerFromQuaternion(self.orientation))

    def update_base_velocity(self):
        """
            Updates the base linear and angular velocity for the current simulation step.
        """
        self.base_linear_velocity, self.base_angular_velocity  = np.array(
            self.p.getBaseVelocity(self.quadruped)
        )

    def update_base_rpy(self):
        """
            Update base orientation (roll, pitch, yaw) for the current simulation step.
        """
        self.base_rpy = np.array(self.p.getEulerFromQuaternion(
            self.p.getBasePositionAndOrientation(self.quadruped)[1] 
        ))

    def update_toes_contact_info(self):
        """
            Updates the contact info for each toe for the current simulation steps. The
            contact info include:
                * terrain_normal_at_each_toe
                * contact_force_at_each_toe
                * foot_ground_friction_coefficients
                * toes_contact_states
        """
        self.terrain_normal_at_each_toe = np.zeros([4, 3])
        self.contact_force_at_each_toe = np.zeros([4])
        self.toes_contact_states = np.zeros([4], dtype=int)
        self.foot_ground_friction_coefficients = np.zeros([4])
        
        for i, toe_id in enumerate(TOES_IDS):
            # Privileged information
            toe_contact_info = self.p.getContactPoints(
                bodyA = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = toe_id
            )

            # No contact case
            if toe_contact_info == (): 
                self.terrain_normal_at_each_toe[i] = (0,0,0)
                self.contact_force_at_each_toe[i] = 0 
                self.foot_ground_friction_coefficients[i] = 0 
                self.toes_contact_states[i] = 0
            else:
                contact_force, fricction_coefficient, normal = \
                    self.__contact_info_average(
                    [ContactInfo(*elem) for elem in (toe_contact_info)]
                )
                self.terrain_normal_at_each_toe[i] = normal
                self.contact_force_at_each_toe[i] = contact_force
                self.foot_ground_friction_coefficients[i] = \
                    fricction_coefficient
                self.toes_contact_states[i] = 1

    def update_thighs_contact_info(self):
        """
            Updates the contact info for each thigh for the current simulation step.
        """
        self.thighs_contact_states = np.zeros([4], dtype=np.int)

        for i, thigh_id in enumerate(THIGHS_IDS):
            thigh_contact_info = self.p.getContactPoints(
                bodyA  = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = thigh_id
            )
            self.thighs_contact_states[i] = int(thigh_contact_info != ())

    def update_shanks_contact_info(self):
        """
            Updates the contact info for each shank for the current simulation step.
        """
        self.shanks_contact_states = np.zeros([4], dtype=np.int)
        for i, shank_id in enumerate(SHANKS_IDS):
            shank_contact_info = self.p.getContactPoints(
                bodyA  = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = shank_id
            )
            self.shanks_contact_states[i] = int(shank_contact_info != ())

    def update_height_scan(self):
        """
            Update the height scan for each step for the current simulation step.
        """
        # 9 scan points around each toe
        self.height_scan_at_each_toe = np.zeros([4, 9]) 
        for i, toe_link_state in enumerate(
            self.p.getLinkStates(self.quadruped, TOES_IDS)
            ):
            toe_link_state =  LinkState(*toe_link_state)
            toe_orientation = toe_link_state.linkWorldOrientation
            toe_position =  toe_link_state.linkWorldPosition
        
            # Height scan around each foot 
            _, _, yaw = self.p.getEulerFromQuaternion(toe_orientation)
            x,y,z =  toe_position 
            P = self.__get_foot_height_scan_coordinates(x,y,yaw) 
            z_terrain = [self.__get_terrain_height(x_p,y_p) for (x_p,y_p) in P]
            self.height_scan_lines[i] = np.array([ 
                [[x, y, z], [x_p, y_p, z_t]] for (x_p, y_p), z_t in zip(P, z_terrain)]
            )
            self.height_scan_at_each_toe[i] = [z_t - z for z_t in z_terrain]

    def update_toes_force(self):
        """
            Update force in each step for the current simulation step.
        """
        toe_force_sensor_threshold = 6      # Newtons 
        self.toe_force_sensor = np.zeros(4) # 4 = Number of toes

        for i, toe_joint_state in enumerate(self.p.getJointStates(
                bodyUniqueId = self.quadruped, 
                jointIndices = TOES_IDS
            )):
            toe_joint_state = JointState(*toe_joint_state) 
            # "Analog" toe force sensor
            F_x, F_y, F_z, _, _, _ = toe_joint_state.jointReactionForces
            F = float(abs(F_x) + abs(F_y) + abs(F_z))
            self.toe_force_sensor[i] = F > toe_force_sensor_threshold

    def update_joints_sensors(self):
        """
            Update position, velocity and torque for each joint for the current
            simulation step.
        """
        # (Position / Velocity / Torque
        # Joint angles
        self.joint_angles = np.zeros(12)      # 12 = Number of DOF // Controlled joints
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)     # For reward calculations   
        
        for i, j_state in enumerate(self.p.getJointStates(
                bodyUniqueId = self.quadruped,
                jointIndices = JOINTS_IDS
            )):
            j_state = JointState(*j_state)
            self.joint_angles[i] = j_state.jointPosition
            self.joint_velocities[i] = j_state.jointVelocity
            self.joint_torques[i] = j_state.appliedJointMotorTorque

    def update_transformation_matrices(self):
        """
            Update the transformation matrices from the hip to the leg base.
        """
        self.transformation_matrices = \
            get_leg_to_horizontal_frame_transformations(self.base_rpy)

    def update_foot_target(self):
        """
            Update the current foot target (position).
        """
        toes_info = self.p.getLinkStates(self.quadruped, TOES_IDS)
        for i, toe_link_state in enumerate(toes_info):
            toe_link_state =  LinkState(*toe_link_state)
            self.foot_target[i] = toe_link_state.linkWorldPosition

    def update_sensor_output(self):
        """
            Updates the sensor states for the current simulation steps.
            It updates the following robot parameters

            Historic Data:
                * joint_position_error_history
            
            Base Velocity:
                * base_linear_velocity
                * base_angular_velocity
            
            Contact info:
                * terrain_normal_at_each_toe
                * contact_force_at_each_toe
                * foot_ground_friction_coefficients
                * toes_contact_states
                * thighs_contact_states
                * shanks_contact_states

            Height Scan:
                * height_scan_at_each_toe
            
            Toe Force Sensors:
                * toe_force_sensor

            Actuated joints sensors:
                * joint_angles
                * joint_velocities
                * joint_torques
        """
        self.update_position_orientation()
        self.update_base_velocity()
        self.update_base_rpy()
        self.update_toes_contact_info()
        self.update_thighs_contact_info()
        self.update_shanks_contact_info()
        self.update_height_scan()
        self.update_toes_force()
        self.update_joints_sensors()
        self.update_transformation_matrices()

    # =========================== DEBUGGING FUNCTIONS =========================== #
    def set_toes_friction_coefficients(self, friction_coefficient: float):
        """
            Changes the friction coeficient of the quadruped toes. It sets the 
            lateral friction coeficient (the one that is mainly used by pybullet)

            Arguments:
            ---------
                friction_coefficient: float
                    The desired friction coeficient to be set on the quadruped toes.
        """
        for toe_id in TOES_IDS:
            self.p.changeDynamics(self.quadruped, toe_id, 
            lateralFriction = friction_coefficient)

    def draw_reference_frame(self, R: np.array, p: np.array, scaling: float=0.05):
        """
            Draws debug lines of a refrence frame represented by the rotation R and 
            the position vector p, in the world frame.

            Arguments:
            ----------
                R: numpy.array, shape (3,3)
                    Rotation matrix
                p: numpy.array, shape (3,) 
                    Position vector
                scaling: float, optional
                    [TODO]
                    Default: 0.05
        """
        # We draw the x axis
        self.p.addUserDebugLine(
            lineFromXYZ = p,
            lineToXYZ = p + R[:,0] * scaling,
            lineColorRGB = [1,0,0],
            lineWidth = 4,
            lifeTime = 0
        )

        # We draw the y axis
        self.p.addUserDebugLine(
            lineFromXYZ = p,
            lineToXYZ = p + R[:,1] * scaling,
            lineColorRGB = [0,1,0],
            lineWidth = 4,
            lifeTime = 0
        )

        # We draw the z axis
        self.p.addUserDebugLine(
            lineFromXYZ = p,
            lineToXYZ = p + R[:,2] * scaling,
            lineColorRGB = [0,0,1],
            lineWidth = 4,
            lifeTime = 0
        )

    def draw_height_field_lines(self):
        """ [TODO] """
        for i, points in enumerate(self.height_scan_lines): # 
            for point in points:
                self.p.addUserDebugLine(point[0], point[1], (0, 1, 0), lifeTime = 3)

    def draw_link_position_lines(self, print_joint_info = False):
        """
            Debugging function used to draw the distance of the quadruped leg 
            links in the scene, and if desired print the joint names and indexs

            It is used to determine the parameters for the inverse kinemtics 
            model.

            It also prints the joint names and indexs using the optional 
            argument print_joint_info.
        
            Arguments:
                ---------
                self: Simulation  ->  Simulation class
                
                print_joint_info  -> bool, default False. If True, prints the
                                        joint names and indexs.
        """
        if print_joint_info:
            _link_name_to_index = \
                {self.p.getBodyInfo(self.quadruped)[0].decode('UTF-8'):-1,}
            for _id in range(self.p.getNumJoints(self.quadruped)):
                _name = \
                    self.p.getJointInfo(self.quadruped, _id)[12].decode('UTF-8')
                _link_name_to_index[_name] = _id
                print("name: " , _name)
                print("index : ", _id)
                print(" ")
        
        n_digits = 5
        
        for i in range(1):
            
            hip_link_state = LinkState(*self.p.getLinkState(self.quadruped,  
            HIPS_IDS[i]))
            
            hip_position = np.array(hip_link_state.worldLinkFramePosition)

            thigh_position = np.array(self.p.getLinkState(self.quadruped,  
            THIGHS_IDS[i])[4])
            
            shank_position = np.array(self.p.getLinkState(self.quadruped,  
            SHANKS_IDS[i])[4])
            
            toe_position = np.array(self.p.getLinkState(self.quadruped,  
            TOES_IDS[i])[4])

            self.p.addUserDebugLine(thigh_position, hip_position,
            (0, 0.5, 0.5), lifeTime = 0)
            self.p.addUserDebugLine(shank_position, thigh_position,
            (0.5, 0.5, 0), lifeTime = 0)
            self.p.addUserDebugLine(toe_position, shank_position, 
            (0, 0, 1), lifeTime = 0)

            offset_hor = abs(hip_position[1] - shank_position[1])
            offset_ver = abs(hip_position[2] - thigh_position[2])
            ls = abs(thigh_position[2] - shank_position[2])
            lw = abs(shank_position[2] - toe_position[2])

            # Horizontal Offset
            self.p.addUserDebugText("h_off =" \
                                    + str(float(offset_hor))[:n_digits], 
            (hip_position + np.array([hip_position[0], shank_position[1],
             hip_position[2]]))/2 , (0, 0, 0), lifeTime = 0)
           
            self.p.addUserDebugLine(hip_position, 
            (hip_position[0], shank_position[1], hip_position[2]),
            (0, 1, 0), lifeTime = 0)

            # Vertical Offset
            self.p.addUserDebugText( "v_off =" +\
                                     str(float(offset_ver))[:n_digits], 
            (np.array([hip_position[0], thigh_position[1], hip_position[2]]) \
            + thigh_position)/2
            , (0, 0, 0), lifeTime = 0)
           
            self.p.addUserDebugLine((hip_position[0], thigh_position[1], 
            hip_position[2]), (thigh_position),(0, 0, 1), lifeTime = 0)

            # Thigh Lenght
            self.p.addUserDebugText("lt = " + str(float(ls))[:n_digits], 
            (thigh_position + np.array([thigh_position[0], thigh_position[1], 
            shank_position[2]]))/2, (0, 0, 0), lifeTime = 0)
            
            self.p.addUserDebugLine(thigh_position, (thigh_position[0], 
            thigh_position[1], shank_position[2]),(1, 0, 0), lifeTime = 0)

            # Shank Lenght Line and text
            self.p.addUserDebugText("ls = " + str(float(lw))[:n_digits], 
            (shank_position + toe_position)/2
            , (0, 0, 0), lifeTime = 0)
            
            self.p.addUserDebugLine(shank_position, (toe_position),
            (0, 1, 0), lifeTime = 0)

            print("hip-thigh distance", np.linalg.norm(hip_position - \
                                                    thigh_position))
            print("thigh-shank distance", np.linalg.norm(thigh_position - \
                                                            shank_position))
            print("shank-toe distance", np.linalg.norm(shank_position - \
                                                            toe_position))

            print("Horizontal Offset : ", offset_hor)
            print("Vertical Offset", offset_ver)
            print("Thigh Lenght :", ls)
            print("Shank Lenght", lw)
            print(np.array(toe_position) - np.array(hip_position))

    def trace_line(self, r_o, r_f, t = 4):    
        """
        Debbuging function to visualize a line between two points in the 
        simulation.

        Arguments
        ---------
        r_o: np.array shape (3,) -> Start point of the line
        r_f: np.array shape (3,) -> End point of the line
        t: float -> default 4. Time in seconds to keep the line visible. 
                            (If set to zero the line wont dissapear)
        Returns
        -------
        None
        """
        self.p.addUserDebugLine(r_o,     
                                r_f,
                                (1, 0, 0), 
                                lifeTime = t)
        
    def meassure_distance(self, r_o, r_f, t=0):
        """
        Auxiliary function to measure the distance between two points in the
        simulation. The functions draws a line between the two points and
        writes the distance in the pybullet simulation.

        Arguments
        ---------
        r_o: np.array shape(,3)-> origin point
        r_f: np.array shape(,3)-> final point
        t: float -> time to wait before measuring the distance. 
                    Default set to 0 (The line doesn't dissaper)
        Returns
        -------
        None
        """
        r_o = np.array(r_o) 
        r_f = np.array(r_f)
        d = np.linalg.norm(r_o - r_f)
        n_digits = 6
        self.p.addUserDebugText("d =" + str(float(d))[:n_digits], 
           (r_o + r_f)/2 , (0, 0, 0), lifeTime = t)
        self.p.addUserDebugLine(r_o,     
                                r_f,
                                (0.7, 0, 0.3), 
                                lifeTime = t)


    # =========================== TESTING FUNCTIONS =========================== #
    def test_sensors(self):
        """ 
        Generates a simulation to test the robot's sensors.
        
        Arguments:
        ---------
            self: Simulation  ->  Simulation class
        """
        t = 0
        while True: 
            self.p.stepSimulation()
            self.update_height_scan()
            time.sleep(1/240)
            t = t+1
            if t % 120 == 0: self.draw_height_field_lines()

    def test_friction(self):
        """
            ESP:
            
            Genera una simulacion para probar el cambio de la friccion entre 
            los pies del robot y el suelo.

            ENG:

            Generates a simulation to test the change of the foot's friction and 
            the ground.
        
        Arguments:
        ---------
            self: Simulation  ->  Simulation class
        
        Returns:
        -------
            None
        """

        t = 0
        friction = 0.4
        while True: 
            self.p.stepSimulation()
            self.update_sensor_output()
            time.sleep(1/240)
            t = t+1
            if t%30 == 0:
                print(self.foot_ground_friction_coefficients)

            if t%120 == 0:
                for toe_id in TOES_IDS:
                    self.p.changeDynamics(self.quadruped, toe_id, 
                                            lateralFriction = friction)
                print('friction = ', friction)
                friction = (friction==0.4)*0.9 + (friction==0.9)*0.1 + \
                                    (friction==0.1)*0.4    
    
    def test_controller_IK(self):
        """
        Test function to test the controller's Inverse Kinematicks

        Arguments
        ----------
        self : -> simulation object
        

        Returns
        -------
        None
        """

        t = 0
        
        self.update_sensor_output()
        hip_position = np.array(self.p.getLinkState(self.quadruped,  
                                HIPS_IDS[0])[4]) 
        toe_position = np.array(self.p.getLinkState(self.quadruped,  
                                TOES_IDS[0])[4])
        print("r_o = ", toe_position - hip_position)
        self.draw_link_position_lines()
        print(self.joint_angles * 180/np.pi)
        x = float(input("x"))
        y = float(input("y"))
        z = float(input("z"))
        objective_position = np.array([x,y,z])
        
        while True: 
            self.p.stepSimulation()
            self.update_sensor_output()
            time.sleep(1/240)
            
            
            t = t+1
            if t%120 == 0:
                for i in range(4):
                    
                    hip_position = np.array(self.p.getLinkState(self.quadruped,  
                                            HIPS_IDS[i])[4]) 
                    self.trace_line(hip_position, 
                                    hip_position + objective_position, t = 0) 
                    r_o = np.array(self.p.getLinkState(self.quadruped,  
                                        TOES_IDS[i])[4])
                    self.meassure_distance(r_o, hip_position + \
                                            objective_position, t=0)
                    self.get_Hi_to_leg_base_transformation_matrices()
                    self.draw_link_position_lines()
                
                print(self.joint_angles * 180/np.pi)
                print(np.array(joint_target_positions) * 180/np.pi)
                x = float(input("x"))
                y = float(input("y"))
                z = float(input("z"))
                objective_position = np.array([x,y,z])
                self.p.removeAllUserDebugItems()
            
            joint_target_positions = []    
            for i in range(4):  
                hip_position = np.array(self.p.getLinkState(self.quadruped,  
                                                    HIPS_IDS[i])[4]) 
  
                if i%2==0: 
                    leg_angles = solve_leg_IK("LEFT", objective_position)                                       
                else:
                    leg_angles = solve_leg_IK("RIGHT", objective_position)
                
                joint_target_positions += list(leg_angles)
            
            self.actuate_joints(joint_target_positions)

    def test_FTG(self, controller):

        """
        Tesitng function to test the controller's Foot Trajectory Generator.

        Arguments
        ----------
        self :-> simulation object
            The simulation object.

        Return
        ------
        None
        """

        t = 0
        # We create a constraint to keep the quadruped over the ground
       
        
        
        sigma_0 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        while True: 
            self.p.stepSimulation()
            self.update_sensor_output()
            time.sleep(1/240) 
            
            # 
            if t%5 == 0:
                joints_angles = []
                nn_output = [0]*16
                target_foot_positions, FTG_frequencies, FTG_phases = \
                        calculate_foot_trajectories(nn_output, t/240,
                                                    sigma_0 = sigma_0,
                                                    f_0=12)
                T_list = \
                    get_leg_to_horizontal_frame_transformations(self.base_rpy)
                
                for i in range(4):
                    r_Hip = np.array(self.p.getLinkState(self.quadruped, 
                            HIPS_IDS[i])[4])
                    
                    r_o = target_foot_positions[i]
                    T = T_list[i]

                    
                    r = T @ np.concatenate((r_o, [1]), axis = 0)
                    r = r[:3]
                    if i%2==0:
                        leg_angles = solve_leg_IK("LEFT", r)
                    else:
                        leg_angles = solve_leg_IK("RIGHT", r)
                    
                    joints_angles += list(leg_angles)

                    # debug
                    self.trace_line(
                                    np.array([r_Hip[0], 
                                    r_Hip[1] +  0.063 * (-1)**i, 
                                    r_Hip[2] - 0.2442]), 
                                    np.array([r_Hip[0], 
                                    r_Hip[1] +  0.063 * (-1)**i, 
                                    r_Hip[2] - 0.2442]) + r_o , 
                                    t =0.2)
                    
                
            self.actuate_joints(joints_angles)
            t = t+1


    
if __name__ == '__main__':
    spot_urdf_file = "../mini_ros/urdf/spot.urdf"
    terrain_file = "../test_terrains/test_terrain.txt" 

    sim = simulation(terrain_file, spot_urdf_file, p,
                self_collision_enabled=False)
    
    sim.initialize(fix_robot_base=True) 

    sim.test_controller_IK()

