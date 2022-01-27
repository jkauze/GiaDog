"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file provides an interface to control and monitor the simulation status.
"""

# Utilities
import time
from typing import *

# Simulacion
import pybullet as p
import pybullet_data as pd
from bullet_dataclasses import *

# Array usage
import numpy as np

class simulation:
    """ Control and monitor the simulation of the spot-mini in pybullet. """
    def __init__(
            self,
            terrain_file: str, 
            giadog_urdf_file: str,
            bullet_server, #Basically the pybullet module
            mesh_scale: List[float]=[1/50, 1/50, 1],
            actuated_joints_ids: List[int] = [7,8,9, 11,12,13, 16,17,18, 20,21,22], 
            thighs_ids: List[int] = [8, 12, 17, 18],
            shanks_ids: List[int] = [9, 13, 18, 22],
            toes_ids: List[int]   = [10, 14, 19, 23],
            gravity_vector: np.ndarray = np.array([0, 0, -9.807])
        ): 
        """
            Arguments:
            ----------
                terrain_file: str
                    Path to the .txt file representing the terrain.
                giadog_urdf_file: str 
                    Path to the URDF file of the quadruped robot.
                bullet_server: module 
                    Pybullet module.
                mesh_scale: List[float], shape (3,), optional 
                    Scaling parameters for the terrain file.
                    Default: [1/50, 1/50, 1]
                actuated_joints_ids: List[int], shape (12,)optional 
                    List with the ids of the quadruped robot actuated joints.
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

                    Default: [7,8,9, 11,12,13, 16,17,18, 20,21,22]
                thighs_ids: List[int], shape (4,), optional
                    Thigs joints id list (The order is below).
                    Default: [8, 12, 17, 18]
                shanks_ids  : List[int], shape (4,), optional
                    Shank joints id list (The order is below).
                    Default: [9, 13, 18, 22]
                toes_ids   : List[int], shape (4,), optional
                    Toes joints id list (The order is below).
                    The order is the following:
                        front_left  (thigh/shak/toe)
                        front_right (thigh/shak/toe)
                        back_left   (thigh/shak/toe)
                        back_right  (thigh/shak/toe)
                    Default: [10, 14, 19, 23]
                gravity_vector: np.ndarray, shape (3,), optional
                    Simulation gravity vector.
                    Default: [0, 0, -9.807]
        """
        self.giadog_urdf_file = giadog_urdf_file
        self.p = bullet_server
        self.mesh_scale = mesh_scale
        self.terrain_file = terrain_file
        
        # This array is used to calculate the robot toes heightfields 
        # Note : The last column is ignored because numpy adds a column of nans while 
        # reading the file
        self.terrain_array = np.genfromtxt(self.terrain_file,  delimiter=",")[:, :-1]
        center_x, center_y = self.terrain_array.shape
        self.center = (center_x // 2, center_y // 2)

        # Robot joint ids
        self.actuated_joints_ids = actuated_joints_ids
        self.thighs_ids = thighs_ids
        self.shanks_ids = shanks_ids
        self.toes_ids = toes_ids
        
        # State data // Sensor data
        self.desired_direction = np.zeros([2])
        self.desired_turning_direction = np.zeros([1])
        self.gravity_vector = gravity_vector
        self.base_linear_velocity = np.zeros([3])
        self.base_angular_velocity = np.zeros([3])
        self.joint_angles = np.zeros([12]) 
        self.joint_velocities = np.zeros([12])

        # FTG (These may be provided by the C++ controller module)
        self.ftg_phases_sin_cos = np.zeros([4,2])
        self.ftg_frequencies = np.zeros([4])
        self.base_frequency = np.zeros([1])

        # Historic data
        self.joint_position_error_history = np.zeros([2, 12])
        self.joint_velocity_history       = np.zeros([2, 12])
        self.foot_target_history          = np.zeros([2, 4, 3])
        
        # Priviledge data
        self.terrain_normal_at_each_toe = np.zeros([4, 3])
        self.normal_force_at_each_toe   = np.zeros([4]) # (Foot contact forces?¿)
        self.toes_contact_states   = np.zeros([4])
        self.thighs_contact_states = np.zeros([4])
        self.shanks_contact_states = np.zeros([4])
        self.height_scan_at_each_toe = np.zeros([4, 9])

        # For debug:
        self.height_scan_lines = np.zeros([4,9,2, 3])
        self.external_force_applied_to_the_base = np.zeros([3])

        # Data only for reward purposes
        self.joint_torques = np.zeros(12)

        # Extra sensor (This may be used in the future)
        self.toe_force_sensor = np.zeros(4) 

    @staticmethod
    def _get_foot_height_scan_coordinates(x: float, y: float, alpha: float) -> np.ndarray:
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
            numpy.ndarray, shape (9, 2)
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
    def _contact_info_average(
            contact_points_info: List[ContactInfo]
            ) -> Tuple[float, float, np.ndarray]: 
        """
            Given a robot toe position and orientation, returns the positions of the toe 
            height sensor coordinates.

            Arguments:
            ----------
            contact_points_info: List[ContactInfo] 
                List containing the contact info of each point that has contact with the
                leg foot.
            
            Returns:
            --------
            float
                magnitude of the normmal force on the foot.
            float
                Friction coeficient between the foot and the terrain.
            np.array, shape (3,)
                direction of the normal force accting on the foot.
        """
        contact_force  = np.array([0,0,0]) 
        friction_force = np.array([0,0,0]) 

        for contact_info in contact_points_info:
            contact_force = contact_force + \
                contact_info.normalForce * np.array(contact_info.contactNormalOnB) 
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

    def initialize(self, x_o: float=0.0, y_o: float=0.0, gui: bool=False):
        """
            Initializes a pybullet simulation, setting up the terrain, gravity and the 
            quadruped in the pybullet enviroment. (And enabiling the torque sensors in the 
            quadruped foots/toes)

            Arguments:
            ----------
            x: float, optional
                x coordintate of the robot initial position (In the world frame).
                Default: 0.0
            y: float, optional
                y coordintate of the robot initial position (In the world frame).
                Default: 0.0
            gui: bool, optional
                Indicates if the simulation GUI will be displayed.
                Default: False
        """
        self.p.connect(self.p.GUI if gui else self.p.DIRECT)

        # Create terrain object
        terrain_shape = self.p.createCollisionShape(
            shapeType = self.p.GEOM_HEIGHTFIELD, 
            meshScale = self.mesh_scale,
            fileName  = self.terrain_file, 
            heightfieldTextureScaling=128
        )
        self.terrain = p.createMultiBody(0, terrain_shape)
        self.p.resetBasePositionAndOrientation(self.terrain, [0,0,0], [0,0,0,1])
        self.p.setGravity(*self.gravity_vector)

        # Get difference between terrain array and real terrain
        ray_info = self.p.rayTest((0, 0, -50),(0, 0, 50))[0]
        self.z_diff = self.terrain_array[self.center[0]][self.center[1]] - ray_info[3][-1]

        # Obtain the maximum height around the starting point
        z_o = -50.0
        x = x_o - 0.2
        while x <= x_o + 0.2:
            y = y_o - 0.2
            while y <= y_o + 0.2:
                z_o = max(z_o, self.get_terrain_height(x, y))
                y += 0.05
            x += 0.05

        # Load mini-spot from URDF file.
        print(f'[i] Initial position: ({x_o}, {y_o}, {z_o})')
        self.quadruped = self.p.loadURDF(
            self.giadog_urdf_file, 
            [x_o, y_o, z_o + 0.3],
            flags = self.p.URDF_USE_SELF_COLLISION | \
                self.p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        )

        # Torque sensors are enable on the quadruped toes
        for toe_id in self.toes_ids:
            self.p.enableJointForceTorqueSensor(
                bodyUniqueId = self.quadruped,
                jointIndex   = toe_id,
                enableSensor = True,
            )

    def get_terrain_height(
            self,
            x: float,
            y: float,
            z_min: float=-50.0,
            z_max: float=50.0
        ) -> float:
        """
            Returns the height of the terrain at x,y coordinates (in cartesian world 
            coordiantes). It needs a position z_o from where to shot a ray that will 
            intersect the terrain. The ray is casted from z_min and z_max to and from the
            z_o, to augment the probabilities of intersection with the terrain, this is
            to avoid  having a ray casted to the robot body. 
            
            This function assumes the terrain has no bridge 'like' structures.

            Arguments:
            ----------
            x: float 
                x position in cartesian global coordinates.
            y: float
                y position in cartesian global coordinates.
            z_min: float, optional 
                Minimum height where the ray is gonna be casted.  
                Should be below the min terrain height.
                Default: -50.0
            z_max: float, optional
                Top height where the ray is gonna be casted.
                Should be above the min terrain height.
                Default: 50.0
            
            Return:
            -------
            float 
                The terrain height at that x, y point.
                If the rays does not intecept the terrain it returns np.NaN
        """
        x = int(x / self.mesh_scale[0]) + self.center[0]
        y = int(y / self.mesh_scale[1]) + self.center[1]

        rows, cols = self.terrain_array.shape
        if x < 0 or x >= rows or y < 0 or y >= cols: return np.NaN

        return self.terrain_array[x][y] - self.z_diff 

    def update_historic_data(self):
        """ 
            Updates the joint position error history for the current simulation step.
        """
        # TODO
        self.joint_position_error_history[1] = self.joint_velocity_history[0]
        #self.joint_position_error_history[0] = self.joint_position_error 
        self.joint_velocity_history[1] = self.joint_velocity_history[0]
        self.joint_velocity_history[0] = self.joint_velocities 
        self.foot_target_history[1]  = self.foot_target_history[0]
        #self.foot_target_history[0]  = self.foot_target

    def update_base_velocity(self):
        """
            Updates the base linear and angular velocity for the current simulation step.
        """
        self.base_linear_velocity, self.base_angular_velocity  = np.array(
            self.p.getBaseVelocity(self.quadruped)
        )

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
        self.toes_contact_states = np.zeros([4], dtype=np.int)
        self.foot_ground_friction_coefficients = np.zeros([4])
        
        for i, toe_id in enumerate(self.toes_ids):
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
                contact_force, fricction_coefficient, normal = self._contact_info_average(
                    [ContactInfo(*elem) for elem in  (toe_contact_info)]
                )
                self.terrain_normal_at_each_toe[i] = normal
                self.contact_force_at_each_toe[i] = contact_force
                self.foot_ground_friction_coefficients[i] = fricction_coefficient
                self.toes_contact_states[i] = 1

    def update_thighs_contact_info(self):
        """
            Updates the contact info for each thigh for the current simulation step.
        """
        self.thighs_contact_states = np.zeros([4], dtype=np.int)

        for i, thigh_id in enumerate(self.thighs_ids):
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
        for i, shank_id in enumerate(self.shanks_ids):
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
            self.p.getLinkStates(self.quadruped, self.toes_ids)
            ):
            toe_link_state =  LinkState(*toe_link_state)
            toe_orientation = toe_link_state.linkWorldOrientation
            toe_position =  toe_link_state.linkWorldPosition
        
            # Height scan around each foot 
            roll, pitch, yaw = self.p.getEulerFromQuaternion(toe_orientation)
            x,y,z =  toe_position 
            P = self._get_foot_height_scan_coordinates(x,y,yaw) 
            z_terrain = [self.get_terrain_height(x_p,y_p,z) for (x_p,y_p) in P]
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
                jointIndices = self.toes_ids
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
        # (Position / Velocity / Torque)    
        # Joint angles
        self.joint_angles = np.zeros(12)      # 12 = Number of DOF // Controlled joints
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)     # For reward calculations   
        
        for i, j_state in enumerate(self.p.getJointStates(
                bodyUniqueId = self.quadruped,
                jointIndices = self.actuated_joints_ids
            )):
            j_state = JointState(*j_state)
            self.joint_angles[i] = j_state.jointPosition
            self.joint_velocities[i] = j_state.jointVelocity
            self.joint_torques[i] = j_state.appliedJointMotorTorque

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
        self.update_historic_data()
        self.update_base_velocity()
        self.update_toes_contact_info()
        self.update_thighs_contact_info()
        self.update_shanks_contact_info()
        self.update_height_scan()
        self.update_toes_force()
        self.update_joints_sensors()

    def actuate_joints(self, joint_target_positions: List[float]):
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

                    Note: It may be useful to add the Kp and Kd as inputs
        """
        self.p.setJointMotorControlArray(
            bodyUniqueId = self.quadruped,
            jointIndices = self.actuated_joints_ids,
            controlMode  = self.p.POSITION_CONTROL,
            targetPositions = joint_target_positions,
        )    

    def draw_height_field_lines(self):
        """ [TODO] """
        for i, points in enumerate(self.height_scan_lines): # 
            for point in points:
                self.p.addUserDebugLine(point[0], point[1], (0, 1, 0), lifeTime = 3)

    def test_sensors(self):
        """ Generate a simulation to test the robot's sensors."""
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
            
            Genera una simulacion para probar el cambio de la friccion en 

            ENG:
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
                for toe_id in self.toes_ids:
                    self.p.changeDynamics(self.quadruped, toe_id, 
                                            lateralFriction = friction)
                print('friction = ', friction)
                friction = (friction==0.4)*0.9 + (friction==0.9)*0.1 + (friction==0.1)*0.4 


    def set_toes_friction_coefficients(self, friction_coefficient):
        """
        Changes the friction coeficient of the quadruped toes. It sets the lateral 
        friction coeficient (the one that is mainly used by pybullet)

        Args:
            self: Simulation  ->  Simulation class

            friction_coefficient: float -> The desired friction coeficient to be 
            set on the quadruped toes.
        """
        for toe_id in self.toes_ids:
            self.p.changeDynamics(self.quadruped, toe_id, 
            lateralFriction = friction_coefficient)

if __name__ == '__main__':
    spot_urdf_file = "../mini_ros/urdf/spot.urdf"
    terrain_file = "../test_terrains/maincra.txt" 

    sim = simulation(terrain_file, spot_urdf_file, p)
    sim.initialize(gui=True) 

    sim.test_sensors()

