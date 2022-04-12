"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file provides an interface to control and monitor the simulation status.
"""

# Utilities
import os
import numpy as np
from time import sleep, time
from typing import List, Tuple, Callable
from terrain_gen import steps, set_goal, save_terrain
from __env__ import EPSILON, MESH_SCALE, GRAVITY_VECTOR, SIM_SECONDS_PER_STEP, \
    TOES_IDS, EXTERNAL_FORCE_MAGN, JOINTS_IDS, THIGHS_IDS, SHANKS_IDS, \
    HIPS_IDS, EXTERNAL_FORCE_TIME, ROWS, COLS, ANGULAR_VEL_NOISE, \
    ORIENTATION_NOISE, VELOCITY_NOISE, ACCELERATION_NOISE, \
    JOINT_ANGLE_NOISE, JOINT_VELOCITY_NOISE

# Simulacion
import pybullet as p
import pybullet_utils.bullet_client as bc
from bullet_dataclasses import ContactInfo, JointState, LinkState
from kinematics import transformation_matrices, solve_leg_IK, \
    rotation_matrix_from_euler, foot_trajectories_debug


class Simulation(object):
    """ Control and monitor the simulation of the spot-mini in pybullet. """

    def __init__(
            self,
            giadog_urdf_file: str,
            gui: bool=False,
            real_step: bool=False,
            self_collision_enabled: bool=False,
        ): 
        """
            Arguments:
            ----------
                giadog_urdf_file: str 
                    Path to the URDF file of the quadruped robot.

                gui: bool, optional
                    Indicates if the simulation GUI will be displayed.
                    Default: False

                real_step: bool, optional

                self_collision_enabled: bool, optional
                    TODO
                    Default: False
        """
        self.giadog_urdf_file = giadog_urdf_file
        self.gui = gui
        self.p = bc.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)

        self.real_step = real_step
        self.initial_time = time() 
        self.timestep = self.initial_time

        if real_step: self.p.setRealTimeSimulation(1)
        else: self.p.setTimeStep(SIM_SECONDS_PER_STEP)

        self.self_collision_enabled = self_collision_enabled
        self.__reset_state()

    def __terrain_height(self, x: float, y: float) -> float:
        """
            Returns the height of the terrain at x,y coordinates (in cartesian 
            world coordiantes). This function assumes the terrain has no bridge 
            'like' structures.

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
        if x == np.NaN or y == np.NaN: return np.NaN
        
        x = int(x / MESH_SCALE[0]) + self.center[0]
        y = int(y / MESH_SCALE[1]) + self.center[1]

        rows, cols = self.terrain_array.shape
        if x < 0 or x >= rows or y < 0 or y >= cols: return np.NaN

        return self.terrain_array[x][y] - self.z_diff 

    def __reset_state(self):
        """
            Reset bot state.
        """
        # Robot state
        self.position    = np.zeros([3])
        self.orientation = np.zeros([3])

        # Non-priviliged data
        self.gravity_vector     = GRAVITY_VECTOR
        # Velocity in world frame
        self.wf_angular_vel     = np.zeros([3])
        self.wf_linear_vel      = np.zeros([3])
        self.wf_linear_vel_prev = np.zeros([3])
        # Acceleration in world frame
        self.angular_vel        = np.zeros([3])
        # Velocity in robot frame
        self.linear_vel         = np.zeros([3])
        self.linear_acc         = np.zeros([3])
        self.joint_angles       = np.zeros([12]) 
        self.joint_velocities   = np.zeros([12])
        self.toes_contact       = np.zeros([4], dtype=np.int8)
        self.thighs_contact     = np.zeros([4], dtype=np.int8)
        self.shanks_contact     = np.zeros([4], dtype=np.int8)
        
        # Priviliged data
        self.normal_toe      = np.zeros([4,3])
        self.height_scan     = np.zeros([4,9])
        self.toes_force1     = np.zeros([4])
        self.toes_force2     = np.zeros(4) 
        self.ground_friction = np.zeros([4])
        self.external_force  = np.zeros([3])

        # Other data
        self.transf_matrices  = np.zeros([4,4,4])
        self.joint_torques    = np.zeros([12])
        self.is_fallen        = False

        # For debug
        self.height_scan_lines = np.zeros([4,9,2,3])

        self.initial_time = time() 
        self.timestep = self.initial_time
        self.dt = 0

    @staticmethod
    def __foot_scan_coordinates(x: float, y: float, alpha: float) -> np.array:
        """
            Given a robot toe position and orientation, returns the positions 
            of the toe height sensor corrdinates.

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

    def __initialize(
            self, 
            x_o: float=0.0, 
            y_o: float=0.0, 
            fix_robot_base: bool=False
        ):
        """
            Initializes a pybullet simulation, setting up the terrain, gravity and the 
            quadruped in the pybullet enviroment. (And enabiling the torque sensors in 
            the quadruped foots/toes)

            Arguments:
            ----------
                x_o: float, optional
                    x coordintate of the robot initial position (In the world 
                    frame).
                    Default: 0.0

                y_o: float, optional
                    y coordintate of the robot initial position (In the world 
                    frame).
                    Default: 0.0

                fix_robot_base: bool, optional
                    [TODO]
                    Default: False
        """
        # Set the gravity vector
        self.p.setGravity(*self.gravity_vector)
        # Create terrain object
        terrain_shape = self.p.createCollisionShape(
            shapeType = self.p.GEOM_HEIGHTFIELD, 
            meshScale = MESH_SCALE,
            fileName  = os.path.realpath(self.terrain_file), 
            heightfieldTextureScaling=128
        )
        self.terrain = self.p.createMultiBody(0, terrain_shape)
        self.p.resetBasePositionAndOrientation(self.terrain, [0,0,0], [0,0,0,1])
        

        # Get difference between terrain array and real terrain
        ray_info = self.p.rayTest((0, 0, -50),(0, 0, 50))[0]
        self.z_diff = self.terrain_array[self.center[0]][self.center[1]] - \
            ray_info[3][-1]

        # Obtain the maximum height around the starting point
        z_o = -50.0
        x = x_o - 0.2
        while x <= x_o + 0.2:
            y = y_o - 0.2
            while y <= y_o + 0.2:
                z_o = max(z_o, self.__terrain_height(x, y))
                y += 0.05
            x += 0.05

        # Load mini-spot from URDF file.
        print(f'\033[1;36m[i]\033[0m Initial position: ({x_o}, {y_o}, {z_o})')
        if self.self_collision_enabled:
            self.quadruped = self.p.loadURDF(
                self.giadog_urdf_file, 
                [x_o, y_o, self.__terrain_height(x_o, y_o) + 0.3],
                flags = self.p.URDF_USE_SELF_COLLISION | \
                   self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
                   useFixedBase = fix_robot_base,
            )
        else:
            self.quadruped = self.p.loadURDF(
                self.giadog_urdf_file, 
                [x_o, y_o, self.__terrain_height(x_o, y_o) + 0.3],
                useFixedBase = fix_robot_base,
            )

        # Torque sensors are enable on the quadruped toes
        for toe_id in TOES_IDS:
            self.p.enableJointForceTorqueSensor(
                bodyUniqueId = self.quadruped,
                jointIndex   = toe_id,
                enableSensor = True,
            )
        
        # Set random external force
        self.__set_external_force()
        self.external_force_applied = False

    def __apply_force(self, F: List[float]):
        """
            Applies a force to the base of the robot.

            Arguments:
            ----------
                F: List[float], shape (3,)
                    Force vector to be applied to the base of the robot.
                    Expressed in the world frame.
        """
        self.p.applyExternalForce(
            self.quadruped, 
            -1, 
            F, 
            [0,0,0], 
            self.p.WORLD_FRAME
        )
    
    def __set_external_force(self):
        """
            Set and randomize the external force applied to the robot base.

            Refrenece:
            ----------
                https://github.com/leggedrobotics/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary/blob/master/include/environment/environment_c010.hpp
                Line: 1575
        """
        # The module is a number sampled from 0 to E  Newtons
        # In the original paper the force is sampled from 0 to 120 Newtons
        force_module = np.random.uniform() * EXTERNAL_FORCE_MAGN # N
        
        # Randomize the direction of the force
        az = np.pi * np.random.uniform()
        el = np.pi/2 * np.random.uniform()

        force_norm = np.array([
            np.cos(az) * np.cos(el),
            np.sin(az) * np.cos(el),
            np.sin(el),
        ])

        self.external_force = force_norm * force_module

    def set_goal(self, x: float, y: float):
        """
            [TODO]
        """
        z = self.__terrain_height(x, y)
        return self.__create_ball(np.array([x,y,z]), 0.1)

    def reset(
            self, 
            terrain_file: str, 
            x_o: float=0.0, 
            y_o: float=0.0,
            fix_robot_base: bool=False
        ):
        """
            Reset simulation.

            Arguments:
            ----------
                terrain_file: str, optional
                    Path to the .txt file representing the terrain.

                x_o: float, optional
                    x coordintate of the robot initial position (In the world 
                    frame).
                    Default: 0.0
                    
                y_o: float, optional
                    y coordintate of the robot initial position (In the world 
                    frame).
                    Default: 0.0
        """
        print(f'\n\033[1;36m[i]\033[0m Restarting simulation.')
        self.terrain_file = terrain_file
        
        # This array is used to calculate the robot toes heightfields 
        # Note : The last column is ignored because numpy adds a column of 
        # nans while reading the file
        self.terrain_array = np.genfromtxt(self.terrain_file,  delimiter=",")
        self.terrain_array = self.terrain_array[:,:-1]
        center_x, center_y = self.terrain_array.shape
        self.center = (center_x // 2, center_y // 2)

        self.p.resetSimulation()
        self.__reset_state()
        self.__initialize(x_o, y_o, fix_robot_base)

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
    @staticmethod
    def __contact_info_average(
            contact_points_info: List[ContactInfo]
        ) -> Tuple[float, float, np.array]: 
        """
            Given a robot toe position and orientation, returns the positions of 
            the toe height sensor coordinates.

            Arguments:
            ----------
                contact_points_info: List[ContactInfo] 
                    List containing the contact info of each point that has 
                    contact with the leg foot.
            
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
        if contact_force_mag != 0:
            fricction_coefficient /= contact_force_mag
            contact_force /= contact_force_mag
        else:
            contact_force  = np.array([0,0,0]) 
            friction_force = np.array([0,0,0])

        return (contact_force_mag, fricction_coefficient, contact_force)

    def __add_noise(self, data: np.array, std: float) -> np.array:
        """
            Add noise to data obtained from a sensor.
            

        """
        return data + np.random.normal(0, std, data.shape)
    
    def step(self):
        """
            Next frame in the simulation.
        """
        if self.real_step:
            self.dt = time() - self.timestep - self.initial_time
        else:
            self.p.stepSimulation()
            self.dt = SIM_SECONDS_PER_STEP
        self.timestep += self.dt

    def update_position_orientation(self):
        """
            [TODO]
        """
        self.position, self.orientation = \
            self.p.getBasePositionAndOrientation(self.quadruped)
        self.orientation = self.p.getEulerFromQuaternion(self.orientation)
        self.orientation = self.__add_noise(
            np.array(self.orientation),
            ORIENTATION_NOISE
        )

    def update_acceleration(self):
        """
            Update the acceleration of the quadruped, by differentiating the 
            velocity (expressed in the worldframe).
        """
        self.wf_linear_vel_prev = self.wf_linear_vel
        self.wf_linear_vel, self.wf_angular_vel  = np.array(
            self.p.getBaseVelocity(self.quadruped)
        )
        self.linear_acc = self.__add_noise(
            (self.wf_linear_vel - self.wf_linear_vel_prev) / SIM_SECONDS_PER_STEP,
            ACCELERATION_NOISE
        )

    def update_base_velocity(self):
        """
            Updates the body linear and angular velocity for the current 
            simulation step.
            
            Applies a transformation matrix to the body linear and angular
            velocities.

            Note: Must be called after updating the acceleration, and the 
            orientation of the quadruped.
        """
        R_world_body = rotation_matrix_from_euler(self.orientation)
        self.linear_vel = self.__add_noise(
            np.dot(R_world_body, self.wf_linear_vel),
            VELOCITY_NOISE * self.dt 
        )
        self.angular_vel = self.__add_noise(
            np.dot(R_world_body, self.wf_angular_vel),
            ANGULAR_VEL_NOISE
        )

    def update_joints_sensors(self):
        """
            Update position, velocity and torque for each joint for the current
            simulation step.
        """ 
        joint_states = self.p.getJointStates(
            bodyUniqueId = self.quadruped,
            jointIndices = JOINTS_IDS
        )
        for i, j_state in enumerate(joint_states):
            j_state = JointState(*j_state)
            self.joint_angles[i]     = self.__add_noise(
                np.array(j_state.jointPosition),
                JOINT_ANGLE_NOISE
            )
            self.joint_velocities[i] = self.__add_noise(
                np.array(j_state.jointVelocity),
                JOINT_VELOCITY_NOISE / (self.dt + EPSILON)
            )
            self.joint_torques[i]    = j_state.appliedJointMotorTorque

    def update_toes_contact_info(self):
        """
            Updates the contact info for each toe for the current simulation steps. The
            contact info include:
                * normal_toe
                * toes_force1
                * ground_friction
                * toes_contact
        """
        for i, toe_id in enumerate(TOES_IDS):
            toe_contact_info = self.p.getContactPoints(
                bodyA = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = toe_id
            )

            if toe_contact_info == (): 
                self.normal_toe[i]      = (0,0,0)
                self.toes_force1[i]     = 0 
                self.ground_friction[i] = 0 
                self.toes_contact[i]    = 0
            else:
                contact_force, fricction_coefficient, normal = \
                    self.__contact_info_average(
                    [ContactInfo(*elem) for elem in (toe_contact_info)]
                )
                self.normal_toe[i]      = normal
                self.toes_force1[i]     = contact_force
                self.ground_friction[i] = fricction_coefficient
                self.toes_contact[i]    = 1

    def update_thighs_contact_info(self):
        """
            Updates the contact info for each thigh for the current simulation step.
        """
        for i, thigh_id in enumerate(THIGHS_IDS):
            thigh_contact_info = self.p.getContactPoints(
                bodyA  = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = thigh_id
            )
            self.thighs_contact[i] = int(thigh_contact_info != ())

    def update_shanks_contact_info(self):
        """
            Updates the contact info for each shank for the current simulation step.
        """
        self.shanks_contact = np.zeros([4], dtype=np.int8)
        for i, shank_id in enumerate(SHANKS_IDS):
            shank_contact_info = self.p.getContactPoints(
                bodyA  = self.quadruped, 
                bodyB = self.terrain, 
                linkIndexA = shank_id
            )
            self.shanks_contact[i] = int(shank_contact_info != ())

    def update_height_scan(self):
        """
            Update the height scan for each step for the current simulation step.
        """
        link_states = self.p.getLinkStates(self.quadruped, TOES_IDS)
        for i, toe_link_state in enumerate(link_states):
            toe_link_state =  LinkState(*toe_link_state)
            toe_orientation = toe_link_state.linkWorldOrientation
            toe_position =  toe_link_state.linkWorldPosition
        
            # Height scan around each foot 
            _, _, yaw = self.p.getEulerFromQuaternion(toe_orientation)
            x,y,z =  toe_position 
            P = self.__foot_scan_coordinates(x,y,yaw) 
            z_terrain = [self.__terrain_height(x_p,y_p) for (x_p,y_p) in P]
            # TODO: a parameter should be put in place to not calculate the scan
            # lines during training.
            self.height_scan_lines[i] = np.array([ 
                [[x, y, z], [x_p, y_p, z_t]] for (x_p, y_p), z_t in zip(P, z_terrain)]
            )
            self.height_scan[i] = [z_t - z for z_t in z_terrain]

    def update_toes_force(self):
        """
            Update force in each step for the current simulation step.
        """
        toe_force_sensor_threshold = 6      # Newtons 
        join_states = self.p.getJointStates(
            bodyUniqueId = self.quadruped, 
            jointIndices = TOES_IDS
        )

        for i, toe_joint_state in enumerate(join_states):
            toe_joint_state = JointState(*toe_joint_state) 
            # "Analog" toe force sensor
            F_x, F_y, F_z, _, _, _ = toe_joint_state.jointReactionForces
            F = float(abs(F_x) + abs(F_y) + abs(F_z))
            self.toes_force2[i] = F > toe_force_sensor_threshold

    def update_external_force(self):
        """
            Update the external force to the base
        """
        if self.timestep < EXTERNAL_FORCE_TIME: 
            self.__apply_force(self.external_force)
        elif not self.external_force_applied: 
            self.external_force = [0, 0, 0]
            self.external_force_applied = True 
            self.__apply_force(self.external_force)
        
    def update_transf_matrices(self):
        """
            Update the transformation matrices from the hip to the leg base.
        """
        self.transf_matrices = transformation_matrices(self.orientation)

    def update_is_fallen(self):
        """
            Update the state that indicates whether the quadruped has fallen.

            If the up directions between the base and the world is larger (the dot
            product is smaller than 0.55), spot is considered fallen.
            
            There was a second condition in the original code, but it was not 
                implemented as it caused early termination of the simulation.
            
            The condition was the following: The base is very low on the ground
            (the height is smaller than 0.13 meter).

            Reference:
            ----------
                Minitaur enviroment (an original pybullet RL enviroment)
        """
        rot_mat = self.p.getMatrixFromQuaternion(
            self.p.getQuaternionFromEuler(self.orientation)
        )
        self.is_fallen = rot_mat[8] < 0.55

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
                * normal_toe
                * toes_force1
                * ground_friction
                * toes_contact
                * thighs_contact
                * shanks_contact

            Height Scan:
                * height_scan
            
            Toe Force Sensors:
                * toe_force_sensor

            Actuated joints sensors:
                * joint_angles
                * joint_velocities
                * joint_torques
        """
        self.update_position_orientation()
        self.update_acceleration()
        self.update_base_velocity()
        self.update_joints_sensors()
        self.update_toes_contact_info()
        self.update_thighs_contact_info()
        self.update_shanks_contact_info()
        self.update_height_scan()
        self.update_toes_force()
        self.update_external_force()
        self.update_transf_matrices()
        self.update_is_fallen()
        

    # ========================= TESTING FUNCTIONS ========================= #
    def __create_vector(
            self, 
            r_o: np.array, 
            r_f: np.array, 
            length: int=1,
            r: int=0, 
            g: int=0, 
            b: int=1
        ) -> int:
        """
            Create a vector between two points in world coordinates.

            Arguments:
            ----------
                r_o: numpy.array, shape (3,)
                    Origin of the vector

                r_f: numpy.array, shape (3,)
                    Final point of the vector

                r: float, optional
                    Red color component.
                    Default: 0

                g: float, optional
                    Green color component.
                    Default: 0

                b: float, optional
                    Blue color component.
                    Default: 1

            Return:
            -------
                Vector id.
        """
        # We get the vector direction
        vector = r_f - r_o
        
        # We get the vector length
        vector_length = np.linalg.norm(vector)
        if vector_length == 0: return -1 

        # We normalize the vector
        vector =  vector / vector_length
        
        # We get the pitch and yaw angles from the vector
        pitch = np.arcsin(-vector[2])
        yaw = np.arctan2(vector[1], vector[0])
        
        thickness = length/400
        # The model of the vector mesures 170 units in the x axis (that explains
        # the scaling for the x axis)
        meshScale=[length/170,thickness,thickness]
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="giadog/assets/vector.obj", rgbaColor=[r,g,b,1], 
            specularColor=[0.4,.4,0], visualFramePosition=[0,0,0],
            meshScale=meshScale
        )

        orientation = p.getQuaternionFromEuler([0, pitch, yaw])
        vector = p.createMultiBody(
            baseMass=0,
            baseOrientation=orientation, 
            baseVisualShapeIndex = visualShapeId, 
            basePosition = r_o, 
            useMaximalCoordinates=False
        )
        
        return vector

    def __update_vector(self, vector_id: int, r_o: np.array, r_f: np.array):
        """
            Update a vector.

            Arguments:
            ----------
                vector_id: int
                    Vector ID.

                r_o: numpy.array, shape (3,)
                    Origin of the vector.

                r_f: numpy.array, shape (3,)
                    Final point of the vector.

            Return:
            -------
                Vector id.
        """
        # We get the vector direction
        vector = r_f - r_o

        norm = np.linalg.norm(vector)
        # Don't draw zero vectors
        if norm == 0: return

        vector = vector / norm
        
        # We get the pitch and yaw angles from the vector
        pitch = np.arcsin(-vector[2])
        yaw = np.arctan2(vector[1], vector[0])

        orientation = self.p.getQuaternionFromEuler([0, pitch, yaw])
        self.p.resetBasePositionAndOrientation(
            vector_id,
            r_o,
            orientation
        )
    
    def __create_ball(
            self,
            r_o : np.array, 
            radius : float,
            r: int=0, 
            g: int=0, 
            b: int=1):
        """
            Creates a visual shape of a ball at position r_o in world 
            coordinates, with the given radius and color.

            Arguments:
            ----------
                r_o: numpy.array, shape (3,)
                    Position of the ball.

                radius: float
                    Radius of the ball.

                r: float, optional
                    Red color component.
                    Default: 0

                g: float, optional
                    Green color component.
                    Default: 0

                b: float, optional
                    Blue color component.
                    Default: 1

            Return:
            -------
                Ball id.
        """
        visualShapeId = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=[r,g,b,1],
            specularColor=[0.4,.4,0],
        )
        ball = p.createMultiBody(
            baseMass=0, 
            baseVisualShapeIndex = visualShapeId, 
            basePosition = r_o, 
            useMaximalCoordinates=False
        )

        return ball
    
    def __update_ball(self, ball_id: int, r_o: np.array):
        """
            Updates the position of a ball.

            Arguments:
            ----------
                ball_id: int
                    Ball ID.

                r_o: numpy.array, shape (3,)
                    Position of the ball.
        """
        p.resetBasePositionAndOrientation(
            ball_id,
            r_o,
            [0,0,0,1]
        )

    def test_desired_direction(self, first_exec: bool=False):
        """
            Test the desired direction by creating a terrain with a random 
            target and generate an arrow that starts from the robot towards 
            said target, and stays that way even while the robot moves.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec:
            self.initial_pos = self.position
            self.initial_orientation = self.orientation

            # Rese state
            self.reset_id = self.p.addUserDebugParameter('RESET', 1, 0, 0)
            self.reset_count = 0

            # Generate a new terrain
            t = steps(500, 500, 1, 0.05, 73)
            x, y = set_goal(t, 2)
            x = x * MESH_SCALE[0] - ROWS * MESH_SCALE[0] / 2
            y = y * MESH_SCALE[1] - COLS * MESH_SCALE[1] / 2
            self.goal = [x, y]
            save_terrain(t, 'terrains/desired_direction_test.txt')
            self.reset('terrains/desired_direction_test.txt')

            self.desired_direction_id = self.__create_vector(
                self.position, 
                np.array(self.goal + [self.position[2]]),
                2,
                *(0, 1, 0)
            )

        direction = np.array(self.goal + [self.position[2]])
        self.__update_vector(
            self.desired_direction_id,
            self.position, 
            direction,
        )

        self.p.resetBasePositionAndOrientation(
            self.quadruped,
            [self.position[0], self.position[1], self.initial_pos[2]],
            self.p.getQuaternionFromEuler(self.initial_orientation)
        )

        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                self.initial_pos,
                self.p.getQuaternionFromEuler(self.initial_orientation)
            )
            self.p.setJointMotorControlArray(
                self.quadruped,
                JOINTS_IDS,
                controlMode=self.p.POSITION_CONTROL,
                targetPositions=[0] * len(JOINTS_IDS)
            )

    def test_position_orientation(self, first_exec: bool=False):
        """
            Test the position and orientation of the robot by creating an arrow that starts from the robot and points where the robot is facing, and stays that way even while the robot is moving.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        r_o = self.position

        # Orientation
        _, pitch, yaw = self.orientation
        x = np.cos(yaw) * np.cos(pitch)
        y = np.sin(yaw) * np.cos(pitch)
        z = -np.sin(pitch)
        r_f = r_o + np.array([x, y, z])

        if first_exec: self.vector_id = self.__create_vector(r_o, r_f)
        else: self.__update_vector(self.vector_id, r_o, r_f)

    def test_linear_velocity_acceleration(self, first_exec: bool=False):
        """
            Test the linear velocity of the robot by forcing it to move 
            horizontally given the velocity parameters given by the user, thus 
            showing the linear velocity arrow as a consequence of the movement.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec: 
            # Force parameters
            self.vel_x_id = self.p.addUserDebugParameter(
                'Velocity X', 
                *(-0.3, 0.3, 0)
            )
            self.vel_y_id = self.p.addUserDebugParameter(
                'Velocity Y', 
                *(-0.3, 0.3, 0)
            )

            # Rese state
            self.reset_id = self.p.addUserDebugParameter('Reset', 1, 0, 0)
            self.reset_count = 0

            # Create position constraint
            self.constraint_id = self.p.createConstraint(
                self.quadruped, 
                -1, -1, -1, 
                self.p.JOINT_FIXED, 
                None, 
                None, 
                [0, 0, 1]
            )
            self.pos_x = 0
            self.pos_y = 0
        else: 
            self.p.removeBody(self.linear_vel_id)
            self.p.removeBody(self.linear_acc_id)

        # Create vectors
        self.linear_vel_id = self.__create_vector(
            self.position, 
            self.position + self.linear_vel,
            np.linalg.norm(self.linear_vel) / 5,
            *(0, 0, 1)
        )
        self.linear_acc_id = self.__create_vector(
            self.position, 
            self.position + self.linear_acc,
            np.linalg.norm(self.linear_acc) / 5,
            *(1, 0, 0)
        )

        # Reset position
        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.pos_x = self.pos_y = 0

        # Update position constraint
        self.p.changeConstraint(
            self.constraint_id,
            [self.pos_x, self.pos_y, 1]
        )

        self.pos_x += self.p.readUserDebugParameter(self.vel_x_id)
        self.pos_y += self.p.readUserDebugParameter(self.vel_y_id)

        # Reset joint positions
        for ID in JOINTS_IDS:
            self.p.resetJointState(self.quadruped, ID, 0)
    

    def test_acceleration_free_fall(self, first_exec: bool=False):
        """
            Test the linear velocity of the robot by forcing it to move 
            horizontally given the velocity parameters given by the user, thus 
            showing the linear velocity arrow as a consequence of the movement.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec: 
        
            # Rese state
            self.reset_id = self.p.addUserDebugParameter('Reset', 1, 0, 0)
            self.reset_count = 0

        else: 
            #self.p.removeBody(self.linear_vel_id)
            self.p.removeBody(self.linear_acc_id)

        # Create vectors
        """
        self.linear_vel_id = self.__create_vector(
            self.position, 
            self.position + self.linear_vel,
            np.linalg.norm(self.linear_vel) / 5,
            *(0, 0, 1)
        )
        """
        self.linear_acc_id = self.__create_vector(
            self.position, 
            self.position + self.linear_acc,
            np.linalg.norm(self.linear_acc) / 25,
            *(1, 0, 0)
        )

        # Reset position
        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.pos_x = self.pos_y = 0

            # Update position constraint

            self.p.resetBasePositionAndOrientation(self.quadruped,
                                                    [0,0,1],
                                                    [0,0,0,1])

        # Reset joint positions
        for ID in JOINTS_IDS:
            self.p.resetJointState(self.quadruped, ID, 0)

    def test_angular_velocity(self, first_exec: bool=False):
        """
            Test the angular velocity of the robot by forcing it to rotate on 
            its own vertical axis given the velocity parameter given by the 
            user, thus showing the angular velocity arrow as a consequence of 
            the movement.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec:
            self.turn = 0

            # Create rotation constraint
            self.constraint_id = self.p.createConstraint(
                self.quadruped, 
                -1, -1, -1, 
                self.p.JOINT_FIXED, 
                None, 
                None, 
                [1, 1, 2],
                self.p.getQuaternionFromEuler([0,0,0])
            )

            # Angular velocity parameter
            self.angular_vel_id = self.p.addUserDebugParameter(
                'Angular velocity', 
                *(-10, 10, 0)
            )
        else:
            self.p.removeBody(self.angular_vector_id)
            self.turn += self.p.readUserDebugParameter(self.angular_vel_id) / 50

        # Create vectors
        self.angular_vector_id = self.__create_vector(
            self.position,
            self.position + self.angular_vel,
            np.linalg.norm(self.angular_vel) / 20,
            *(0, 0, 1)
        )

        # Update rotation constraint
        self.p.changeConstraint(
            self.constraint_id,
            jointChildFrameOrientation=self.p.getQuaternionFromEuler(
                [0, 0, self.turn]
            )
        )

    def test_joint_sensors(self, first_exec: bool=False):
        """
            Allows the user to modify the rotation of the hip, upper leg and 
            lower leg joints of the upper right leg of the robot, to then 
            constantly display the angle, angular velocity and torque of 
            those 3 joints and test that these data correspond to the 
            simulation.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec: 
            self.initial_pos = self.position
            self.initial_orientation = self.orientation

            # Velocities parameters
            self.vel1_id = self.p.addUserDebugParameter(
                'Hip velocity', 
                *(-10, 10, 0)
            )
            self.vel2_id = self.p.addUserDebugParameter(
                'Upper leg velocity', 
                *(-10, 10, 0)
            )
            self.vel3_id = self.p.addUserDebugParameter(
                'Lower leg velocity', 
                *(-10, 10, 0)
            )

            self.reset_id = self.p.addUserDebugParameter('Reset', 1, 0, 0)
            self.reset_count = 0

        self.p.resetBasePositionAndOrientation(
            self.quadruped,
            self.initial_pos,
            self.p.getQuaternionFromEuler(self.initial_orientation)
        )

        # Reset position
        if self.reset_count == self.p.readUserDebugParameter(self.reset_id):
            self.p.setJointMotorControlArray(
                self.quadruped,
                JOINTS_IDS[3:6],
                controlMode=self.p.VELOCITY_CONTROL,
                targetVelocities=[
                    self.p.readUserDebugParameter(self.vel1_id),
                    self.p.readUserDebugParameter(self.vel2_id),
                    self.p.readUserDebugParameter(self.vel3_id),
                ]
            )
        else:
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.p.resetJointState(self.quadruped, JOINTS_IDS[3], 0)
            self.p.resetJointState(self.quadruped, JOINTS_IDS[4], 0)
            self.p.resetJointState(self.quadruped, JOINTS_IDS[5], 0)

        static_joints = JOINTS_IDS[:3] + JOINTS_IDS[6:]
        self.p.setJointMotorControlArray(
            self.quadruped,
            static_joints,
            controlMode=self.p.POSITION_CONTROL,
            targetPositions=[0] * len(static_joints)
        )

        print(
            'HIP JOINT ANGLE: {:.4f} | '.format(self.joint_angles[3]) +\
            'UPPER LEG JOINT ANGLE: {:.4f} | '.format(self.joint_angles[4]) +\
            'LOWER LEG JOINT ANGLE: {:.4f}\n'.format(self.joint_angles[5]) +\

            'HIP JOINT VELOCITY: {:.4f} | '.format(self.joint_velocities[3]) +\
            'UPPER LEG JOINT VELOCITY: {:.4f} | '.format(self.joint_velocities[4]) +\
            'LOWER LEG JOINT VELOCITY: {:.4f}\n'.format(self.joint_velocities[5]) +\

            'HIP JOINT TORQUE: {:.4f} | '.format(self.joint_torques[3]) +\
            'UPPER LEG JOINT TORQUE: {:.4f} | '.format(self.joint_torques[4]) +\
            'LOWER LEG JOINT TORQUE: {:.4f}\n\n'.format(self.joint_torques[5]) 
        )

    def test_toes_contact(self, first_exec: bool=False):
        """
            Tests the contact sensors of the robot's feet by allowing the user 
            to move the robot up and down, and constantly printing in the 
            terminal the array that indicates if each foot made contact with 
            the ground in addition to showing in the simulation the vector
            that represents the normal force on each leg

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec:
            self.toes_force_id = [-1] * 4
            self.initial_pos = self.position
            self.initial_orientation = self.orientation

            self.go_up_id = self.p.addUserDebugParameter('GO UP', 1, 0, 0)
            self.go_up_count = 0
            self.current_go_up = False

            self.go_down_id = self.p.addUserDebugParameter('GO DOWN', 1, 0, 0)
            self.go_down_count = 0
            self.current_go_down = False

            self.reset_id = self.p.addUserDebugParameter('RESET', 1, 0, 0)
            self.reset_count = 0

            for i, data in enumerate(self.p.getLinkStates(self.quadruped, TOES_IDS)):
                pos = LinkState(*data).linkWorldPosition
                self.toes_force_id[i] = self.__create_vector(
                    pos, 
                    np.ones((3,)),
                    0.3,
                    *(0, 0, 1)
                )

        # Update normal vectors
        for i, data in enumerate(self.p.getLinkStates(self.quadruped, TOES_IDS)):
            if self.toes_contact[i]:
                pos = LinkState(*data).linkWorldPosition
                self.__update_vector(
                    self.toes_force_id[i],
                    pos, 
                    pos + self.normal_toe[i],
                )
            else:
                # If there is no contact, the vectors are drawn away so that 
                # they are not seen.
                self.__update_vector(
                    self.toes_force_id[i],
                    np.array([-100, -100, -100]), 
                    np.array([-101, -101, -101]),
                )

        # Verify buttons
        if self.go_up_count != self.p.readUserDebugParameter(self.go_up_id):
            self.go_up_count = self.p.readUserDebugParameter(self.go_up_id)
            self.current_go_up = True
            self.current_go_down = False

        if self.go_down_count != self.p.readUserDebugParameter(self.go_down_id):
            self.go_down_count = self.p.readUserDebugParameter(self.go_down_id)
            self.current_go_up = False
            self.current_go_down = True

        # Update position
        self.p.resetBasePositionAndOrientation(
            self.quadruped,
            [self.initial_pos[0], self.initial_pos[1], self.position[2]],
            self.p.getQuaternionFromEuler(self.initial_orientation)
        )

        # Reset robot and joints positions
        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                self.initial_pos,
                self.p.getQuaternionFromEuler(self.initial_orientation)
            )
            for ID in JOINTS_IDS:
                self.p.resetJointState(self.quadruped, ID, 0)

        # Apply forces
        if self.current_go_up: self.__apply_force([0, 0, 150])
        elif self.current_go_down: self.__apply_force([0, 0, -50])

        print(f'TOES CONTACT: {self.toes_contact}')

    def test_thighs_shanks_contact(self, first_exec: bool=False):
        """
            Places the robot inside a box, then allows the user to move it 
            horizontally, being able to collide with the walls of the box. 
            The terminal is constantly showing the arrays that indicates if 
            the thighs or the shanks came into contact.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec: 
            self.initial_pos = self.position
            self.initial_orientation = self.orientation

            # Move parameters
            self.north_id = self.p.addUserDebugParameter('NORTH', 1, 0, 0)
            self.north_count = 0
            self.go_north = False

            self.east_id = self.p.addUserDebugParameter('EAST', 1, 0, 0)
            self.east_count = 0
            self.go_east = False

            self.south_id = self.p.addUserDebugParameter('SOUTH', 1, 0, 0)
            self.south_count = 0
            self.go_south = False

            self.west_id = self.p.addUserDebugParameter('WEST', 1, 0, 0)
            self.west_count = 0
            self.go_west = False

            self.stop_id = self.p.addUserDebugParameter('STOP', 1, 0, 0)
            self.stop_count = 0

            # Rese state
            self.reset_id = self.p.addUserDebugParameter('RESET', 1, 0, 0)
            self.reset_count = 0

        # Verify buttons
        if self.north_count != self.p.readUserDebugParameter(self.north_id):
            self.north_count = self.p.readUserDebugParameter(self.north_id)
            self.go_north = True
            self.go_east = False
            self.go_south = False 
            self.go_west = False 

        if self.east_count != self.p.readUserDebugParameter(self.east_id):
            self.east_count = self.p.readUserDebugParameter(self.east_id)
            self.go_north = False
            self.go_east = True
            self.go_south = False 
            self.go_west = False 

        if self.south_count != self.p.readUserDebugParameter(self.south_id):
            self.south_count = self.p.readUserDebugParameter(self.south_id)
            self.go_north = False
            self.go_east = False
            self.go_south = True 
            self.go_west = False 

        if self.west_count != self.p.readUserDebugParameter(self.west_id):
            self.west_count = self.p.readUserDebugParameter(self.west_id)
            self.go_north = False
            self.go_east = False
            self.go_south = False 
            self.go_west = True 

        if self.stop_count != self.p.readUserDebugParameter(self.stop_id):
            self.stop_count = self.p.readUserDebugParameter(self.stop_id)
            self.go_north = False
            self.go_east = False
            self.go_south = False 
            self.go_west = False 

        # Reset position
        if self.reset_count == self.p.readUserDebugParameter(self.reset_id):
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                [self.position[0], self.position[1], self.initial_pos[2]],
                self.p.getQuaternionFromEuler(self.initial_orientation)
            )
        else:
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                self.initial_pos,
                self.p.getQuaternionFromEuler(self.initial_orientation)
            )
            for ID in JOINTS_IDS:
                self.p.resetJointState(self.quadruped, ID, 0)

        if self.go_north: self.__apply_force([-200, 0, 0])
        elif self.go_east: self.__apply_force([0, 200, 0])
        elif self.go_south: self.__apply_force([200, 0, 0])
        elif self.go_west: self.__apply_force([0, -200, 0])

        print(
            f'THIGHS CONTACTS: {self.thighs_contact} | ' +\
            f'SHANKS CONTACTS {self.shanks_contact}'
        )

    def test_friction(self, first_exec: bool=False):
        """
            It places the robot on a fairly rough terrain, then allows the 
            user to move it both horizontally and vertically and constantly 
            prints through the terminal the friction force that the legs 
            receive when in contact with the ground.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        if first_exec:
            self.toes_force_id = [-1] * 4
            self.initial_pos = np.array(self.position)
            self.current_pos = np.array(self.position)
            self.initial_orientation = self.orientation

            # Move parameters
            self.north_id = self.p.addUserDebugParameter('NORTH', 1, 0, 0)
            self.north_count = 0

            self.east_id = self.p.addUserDebugParameter('EAST', 1, 0, 0)
            self.east_count = 0

            self.south_id = self.p.addUserDebugParameter('SOUTH', 1, 0, 0)
            self.south_count = 0

            self.west_id = self.p.addUserDebugParameter('WEST', 1, 0, 0)
            self.west_count = 0

            self.go_up_id = self.p.addUserDebugParameter('GO UP', 1, 0, 0)
            self.go_up_count = 0

            self.go_down_id = self.p.addUserDebugParameter('GO DOWN', 1, 0, 0)
            self.go_down_count = 0

            self.stop_id = self.p.addUserDebugParameter('STOP', 1, 0, 0)
            self.stop_count = 0
            self.current_state = 'STOP'

            self.reset_id = self.p.addUserDebugParameter('RESET', 1, 0, 0)
            self.reset_count = 0

        # Verify buttons
        if self.north_count != self.p.readUserDebugParameter(self.north_id):
            self.north_count = self.p.readUserDebugParameter(self.north_id)
            self.current_state = 'NORTH'

        if self.east_count != self.p.readUserDebugParameter(self.east_id):
            self.east_count = self.p.readUserDebugParameter(self.east_id)
            self.current_state = 'EAST'

        if self.south_count != self.p.readUserDebugParameter(self.south_id):
            self.south_count = self.p.readUserDebugParameter(self.south_id)
            self.current_state = 'SOUTH'

        if self.west_count != self.p.readUserDebugParameter(self.west_id):
            self.west_count = self.p.readUserDebugParameter(self.west_id)
            self.current_state = 'WEST'

        if self.go_up_count != self.p.readUserDebugParameter(self.go_up_id):
            self.go_up_count = self.p.readUserDebugParameter(self.go_up_id)
            self.current_state = 'GO UP'

        if self.go_down_count != self.p.readUserDebugParameter(self.go_down_id):
            self.go_down_count = self.p.readUserDebugParameter(self.go_down_id)
            self.current_state = 'GO DOWN'

        if self.stop_count != self.p.readUserDebugParameter(self.stop_id):
            self.stop_count = self.p.readUserDebugParameter(self.stop_id)
            self.current_state = 'STOP'

        # Reset
        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.current_pos = np.array(self.initial_pos)
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                self.initial_pos,
                self.p.getQuaternionFromEuler(self.initial_orientation)
            )
        # Horizontal move
        elif self.current_state in ['NORTH', 'EAST', 'SOUTH', 'WEST']:
            self.current_pos[:2] = self.position[:2]
        # Vertical move 
        elif self.current_state in ['GO UP', 'GO DOWN']:
            self.current_pos[2] = self.position[2]

        self.p.resetBasePositionAndOrientation(
            self.quadruped,
            self.current_pos,
            self.p.getQuaternionFromEuler(self.initial_orientation)
        )
        for ID in JOINTS_IDS:
            self.p.resetJointState(self.quadruped, ID, 0)

        # Apply forces
        if self.current_state == 'NORTH': self.__apply_force([-200, 0, 0])
        elif self.current_state == 'EAST': self.__apply_force([0, 200, 0])
        elif self.current_state == 'SOUTH': self.__apply_force([200, 0, 0])
        elif self.current_state == 'WEST': self.__apply_force([0, -200, 0])
        elif self.current_state == 'GO UP': self.__apply_force([0, 0, 100])
        elif self.current_state == 'GO DOWN': self.__apply_force([0, 0, -70])

        print('GROUND FRICTION: [{:.4f}, {:.4f}, {:.4f}, {:.4f}]'.format(*self.ground_friction))

    def test_height_scan(self, first_exec: bool=False):
        """
            Tests the height scan of the robot, by drawing a ball in the scaned 
            point on a fairly irregular steps terrain.

            Arguments:
            ----------
                first_exec: bool
                    if True, the parameters are initialized.
        """
        
        if first_exec:
            # Force parameters
            self.vel_x_id = self.p.addUserDebugParameter(
                'Velocity X', 
                *(-1, 1, 0)
            )
            self.vel_y_id = self.p.addUserDebugParameter(
                'Velocity Y', 
                *(-1, 1, 0)
            )

            # Rese state
            self.reset_id = self.p.addUserDebugParameter('Reset', 1, 0, 0)
            self.reset_count = 0

            # Create position constraint
            self.constraint_id = self.p.createConstraint(
                self.quadruped, 
                -1, -1, -1, 
                self.p.JOINT_FIXED, 
                None, 
                None, 
                [0, 0, 0.5]
            )
            self.pos_x = 0
            self.pos_y = 0

            self.balls = []
            for i, points in enumerate(self.height_scan_lines): 
                balls_i = []
                for point in points:
                    balls_i.append(self.__create_ball(
                        point[1],
                        0.015,
                    ))
                self.balls.append(balls_i)

        # Reset position
        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            self.pos_x = self.pos_y = 0
            # Reset joint positions
            for ID in JOINTS_IDS:
                self.p.resetJointState(self.quadruped, ID, 0)

        # Update position constraint
        self.p.changeConstraint(
            self.constraint_id,
            [self.pos_x, self.pos_y, 0.5]
        )

        self.pos_x += self.p.readUserDebugParameter(self.vel_x_id) / 200
        self.pos_y += self.p.readUserDebugParameter(self.vel_y_id) / 200

        for i, points in enumerate(self.height_scan_lines): 
            for j, point in enumerate(points):
                self.__update_ball(
                    self.balls[i][j],
                    point[1]
                )

    def test_FTG(self, first_exec: bool=False):

        """
        Tesitng function to test the controller's Foot Trajectory Generator.

        Arguments
        ----------
        first_exec: bool -> if True, the parameters are initialized.

        Return
        ------
        None
        """

        # We create a constraint to keep the quadruped over the ground
        if first_exec:
            # Force parameters
            self.phase_leg_1 = self.p.addUserDebugParameter(
                'Leg 1', 
                *(0, 2*np.pi, 0)
            )
            self.phase_leg_2 = self.p.addUserDebugParameter(
                'Leg 2', 
                 *(0, 2*np.pi, 0)
            )
            self.phase_leg_3 = self.p.addUserDebugParameter(
                'Leg 3', 
                *(0, 2*np.pi, 0)
            )
            self.phase_leg_4 = self.p.addUserDebugParameter(
                'Leg 4', 
                 *(0, 2*np.pi, 0)
            )
            self.base_frequency = self.p.addUserDebugParameter(
                'Base frequency',
                *(0, 16, 2.5))

            self.reset_id = self.p.addUserDebugParameter('Reset', 1, 0, 0)
            self.reset_count = 0
            self.t = 0
        
        sigma_0 = np.array([self.p.readUserDebugParameter(self.phase_leg_1), 
                            self.p.readUserDebugParameter(self.phase_leg_2), 
                            self.p.readUserDebugParameter(self.phase_leg_3), 
                            self.p.readUserDebugParameter(self.phase_leg_4)])
        base_frequency = self.p.readUserDebugParameter(self.base_frequency)
        line_colors = [
            (0,0,1),#blue
            (0,1,0),#green
            (1,0,0),#red
            (1,1,1),#white
        ] 

        if self.reset_count != self.p.readUserDebugParameter(self.reset_id):
            self.reset_count = self.p.readUserDebugParameter(self.reset_id)
            # Reset robot position
            self.p.resetBasePositionAndOrientation(
                self.quadruped,
                (0,0,0.5),
                (0,0,0,1)
            )
            
            
        self.joints_angles = []
        nn_output = [0]*16
        target_foot_positions, FTG_frequencies, FTG_phases = \
                foot_trajectories_debug(nn_output, self.timestep,
                                            sigma_0 = sigma_0,
                                            f_0=base_frequency)
        T_list = transformation_matrices(self.orientation)
        
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
            
            self.joints_angles += list(leg_angles)

            
            # Trace lines where the feet are
            self.trace_line(
                            np.array([r_Hip[0], 
                            r_Hip[1] +  0.063 * (-1)**i, 
                            r_Hip[2] - 0.2442]), 
                            np.array([r_Hip[0], 
                            r_Hip[1] +  0.063 * (-1)**i, 
                            r_Hip[2] - 0.2442]) + r_o , 
                            t =40,
                            color = line_colors[i])
            
        self.actuate_joints(self.joints_angles)

    def test_IK(self, first_exec: bool=False):
        """
        
        """

        if first_exec:
            # We draw the H_i frames below the robot hips
            for i in range(4):
                r_Hip = np.array(self.p.getLinkState(self.quadruped,
                    HIPS_IDS[i])[4]) 
                pos = np.array([r_Hip[0], 
                                r_Hip[1] +  0.063 * (-1)**i, 
                                r_Hip[2] - 0.2442])
                Rot = np.eye(3)
                self.draw_reference_frame(Rot, pos)
        
            self.x_pos_param = self.p.addUserDebugParameter(
                'x', 
                *(-0.2, 0.2, 0)
            )
            self.y_pos_param = self.p.addUserDebugParameter(
                'y', 
                 *(-0.2, 0.2, 0)
            )
            self.z_pos_param = self.p.addUserDebugParameter(
                'z', 
                *(-0.2, 0.2, 0)
            )

            self.leg_1 = self.p.addUserDebugParameter('Leg 1', 1, 0, 0)
            self.leg_2 = self.p.addUserDebugParameter('Leg 2', 1, 0, 0)
            self.leg_3 = self.p.addUserDebugParameter('Leg 3', 1, 0, 0)
            self.leg_4 = self.p.addUserDebugParameter('Leg 4', 1, 0, 0)
            self.leg_1_count = 0
            self.leg_2_count = 0
            self.leg_3_count = 0
            self.leg_4_count = 0

            self.objective_joint_angles = self.joint_angles
    
        x_pos = self.p.readUserDebugParameter(self.x_pos_param)
        y_pos = self.p.readUserDebugParameter(self.y_pos_param)
        z_pos = self.p.readUserDebugParameter(self.z_pos_param)

        r = np.array([x_pos, y_pos, z_pos])

        T_list = self.transf_matrices
        
        if self.leg_1_count != self.p.readUserDebugParameter(self.leg_1):
            self.objective_joint_angles = self.joint_angles.copy()
            T = T_list[0]
            r = T @ np.concatenate((r, [1]), axis = 0)
            r = r[:3]
            leg_angles = solve_leg_IK("LEFT", r)
            self.leg_1_count = self.p.readUserDebugParameter(self.leg_1)
            self.objective_joint_angles[0:3] = leg_angles
        
        if self.leg_2_count != self.p.readUserDebugParameter(self.leg_2):
            self.objective_joint_angles = self.joint_angles.copy()
            T = T_list[1]
            r = T @ np.concatenate((r, [1]), axis = 0)
            r = r[:3]
            leg_angles = solve_leg_IK("RIGHT", r)
            self.objective_joint_angles[3:6] = leg_angles
            self.leg_2_count = self.p.readUserDebugParameter(self.leg_2)
        
        if self.leg_3_count != self.p.readUserDebugParameter(self.leg_3):
            self.objective_joint_angles = self.joint_angles.copy()
            T = T_list[2]
            r = T @ np.concatenate((r, [1]), axis = 0)
            r = r[:3]
            leg_angles = solve_leg_IK("LEFT", r)
            self.objective_joint_angles[6:9] = leg_angles
            self.leg_3_count = self.p.readUserDebugParameter(self.leg_3)

        if self.leg_4_count != self.p.readUserDebugParameter(self.leg_4):
            self.objective_joint_angles = self.joint_angles.copy()
            T = T_list[3]
            r = T @ np.concatenate((r, [1]), axis = 0)
            r = r[:3]
            leg_angles = solve_leg_IK("RIGHT", r)
            self.objective_joint_angles[9:12] = leg_angles
            self.leg_4_count = self.p.readUserDebugParameter(self.leg_4)


        self.actuate_joints(self.objective_joint_angles)

    def test(self, test_function: Callable):
        """
            Function to run a test.
            
            Note: The simulation is runned at real time.

            Arguments:
            ----------
                test_function: Callable
                    Test function to run, each timestep.
        """
        # Update simulation
        self.step()
        self.update_sensor_output()
        test_function(True)

        while True:
            self.step()
            self.update_sensor_output()
            test_function(False)

    # ========================= DEBUGGING FUNCTIONS ========================= #
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
            Draws debug lines of a refrence frame represented by the rotation R 
            and the position vector p, in the world frame.

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

    def trace_line(self, r_o, r_f, t = 4, color = (1, 0, 0)):    
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
                                color, 
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


    # =========================== TESTING FUNCTIONS ========================== #
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
            sleep(1/240)
            t = t+1
            if t % 120 == 0: self.draw_height_field_lines()

    def test_friction_(self):
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
            sleep(1/240)
            t = t+1
            if t%30 == 0:
                print(self.ground_friction)

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
            sleep(1/240)
            
            
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

