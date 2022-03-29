"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file provides an interface to control and monitor the simulation status.
"""

# Utilities
import numpy as np
from typing import *
from time import sleep
from __env__ import MESH_SCALE, GRAVITY_VECTOR, SIM_SECONDS_PER_STEP, \
    TOES_IDS, EXTERNAL_FORCE_MAGN, JOINTS_IDS, THIGHS_IDS, SHANKS_IDS, \
    HIPS_IDS, EXTERNAL_FORCE_TIME

# Simulacion
import pybullet as p
from kinematics import *
from bullet_dataclasses import *
import pybullet_utils.bullet_client as bc


class Simulation(object):
    """ Control and monitor the simulation of the spot-mini in pybullet. """
    def __init__(
            self,
            giadog_urdf_file: str,
            gui: bool=False,
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

                self_collision_enabled: bool, optional
                    TODO
                    Default: False
        """
        self.giadog_urdf_file = giadog_urdf_file
        self.gui = gui
        self.p = bc.BulletClient(connection_mode=p.GUI if gui else p.DIRECT)
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
        self.angular_vel        = np.zeros([3])
        self.linear_vel         = np.zeros([3])
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
        self.base_rpy         = np.zeros([3]) # TODO
        self.transf_matrices  = np.zeros([4,4,4])
        self.joint_torques    = np.zeros([12])
        self.is_fallen        = False

        # For debug
        self.height_scan_lines = np.zeros([4,9,2,3])

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
        # Create terrain object
        terrain_shape = self.p.createCollisionShape(
            shapeType = self.p.GEOM_HEIGHTFIELD, 
            meshScale = MESH_SCALE,
            fileName  = os.path.realpath(self.terrain_file), 
            heightfieldTextureScaling=128
        )
        self.terrain = self.p.createMultiBody(0, terrain_shape)
        self.p.resetBasePositionAndOrientation(self.terrain, [0,0,0], [0,0,0,1])
        self.p.setGravity(*self.gravity_vector)

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
        self.timestep = 0.0

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

    def reset(self, terrain_file: str, x_o: float=0.0, y_o: float=0.0):
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

    def step(self):
        """
            Next frame in the simulation.
        """
        self.p.stepSimulation()
        self.timestep += SIM_SECONDS_PER_STEP

    def update_position_orientation(self):
        """
            [TODO]
        """
        self.position, self.orientation = \
            self.p.getBasePositionAndOrientation(self.quadruped)
        self.orientation = self.p.getEulerFromQuaternion(self.orientation)
        self.orientation = np.array(self.orientation)

    def update_base_velocity(self):
        """
            Updates the base linear and angular velocity for the current simulation step.
        """
        self.linear_vel, self.angular_vel  = np.array(
            self.p.getBaseVelocity(self.quadruped)
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
            self.joint_angles[i]     = j_state.jointPosition
            self.joint_velocities[i] = j_state.jointVelocity
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

    def update_base_rpy(self):
        """
            Update base orientation (roll, pitch, yaw) for the current simulation step.
        """
        self.base_rpy = np.array(self.p.getEulerFromQuaternion(
            self.p.getBasePositionAndOrientation(self.quadruped)[1] 
        ))
        
    def update_transf_matrices(self):
        """
            Update the transformation matrices from the hip to the leg base.
        """
        self.transf_matrices = transformations_matrices(self.base_rpy)

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
        self.update_base_velocity()
        self.update_joints_sensors()
        self.update_toes_contact_info()
        self.update_thighs_contact_info()
        self.update_shanks_contact_info()
        self.update_height_scan()
        self.update_toes_force()
        self.update_external_force()
        self.update_base_rpy()
        self.update_transf_matrices()
        self.update_is_fallen()
    

    def draw_vector(self, r_o, r_f, r = 0, g= 0, b = 1):
        """
        Draw a vector between two points in world coordinates.

        Arguments:
        ----------

        r_o: (3,) numpy array :-> origin of the vector
        r_f: (3,) numpy array :-> final point of the vector
        r: float :-> red color component
        g: float :-> green color component
        b: float :-> blue color component
        """
        # We get the vecor direction
        vector = r_f - r_o
        
        # We get the vector length
        vector_length = np.linalg.norm(vector)
        # We normalize the vector
        vector = vector / vector_length
        
        # We get the pitch and yaw angles from the vector
        pitch = np.arcsin(-vector[2])
        yaw = np.arctan2(vector[1], vector[0])
        
        thickness = vector_length/400
        # The model of the vector mesures 170 units in the x axis (that explains
        # the scaling for the x axis)
        meshScale=[vector_length/170,thickness,thickness]
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                    fileName="giadog/assets/vector.obj", rgbaColor=[r,g,b,1], 
                    specularColor=[0.4,.4,0], visualFramePosition=[0,0,0],
                    meshScale=meshScale)

                                
        orientation = p.getQuaternionFromEuler([0,pitch,yaw])
        vector = p.createMultiBody(baseMass=0,
                                baseOrientation=orientation, 
                                baseVisualShapeIndex = visualShapeId, 
                                basePosition = r_o, 
                                useMaximalCoordinates=False)
        
        return vector

    # ========================= TEST FUNCTIONS ========================= #
    def test_position_orientation(self):
        """
            [TODO]
        """
        # Position
        r_o = self.position
        # Orientation
        _, pitch, yaw = self.orientation
        x = np.cos(yaw) * np.cos(pitch)
        y = np.sin(yaw) * np.cos(pitch)
        z = np.sin(pitch)
        r_f = r_o + np.array([x, y, z])

        self.trace_line(r_o, r_f, 0.1)

    def test(self, test_function: Callable):
        """
            [TODO]
        """
        while True:
            self.step()
            self.update_sensor_output()
            test_function()

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
            sleep(1/240)
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

    def test_FTG(self):

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
            sleep(1/240) 
            
            # 
            if t%5 == 0:
                joints_angles = []
                nn_output = [0]*16
                target_foot_positions, FTG_frequencies, FTG_phases = \
                        foot_trajectories(nn_output, t/240,
                                                    sigma_0 = sigma_0,
                                                    f_0=12)
                T_list = transformations_matrices(self.base_rpy)
                
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


