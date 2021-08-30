"""
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/08/13

	[TODO: DESCRIPTION]
"""
# Utilidades
from typing import *
from sys import argv, stderr
import time

# Simulacion
import pybullet as p
import pybullet_data as pd

# Array usage
import numpy as np
from scipy.interpolate import interp2d

#pybullet Dataclases
from bullet_dataclasses import *

# Maybe this should be in a separate file (Auxiliary function)
def get_foot_height_scan_coordinates(x,y, alpha):
	"""
		Given a robot toe position and orientation, returns the positions of the toe 
		height sensor corrdinates.

		Args:
			x: float  ->  x coordinate of the robot toe. [In the world frame]
			y: float  ->  y coordinate of the robot toe. [In the world frame]
			alpha: float  ->  Orientation of the toe.
		Return:
			np.array -> q(2, 9) numpy array with each of the toe height sensor 
			corrdinates
	"""
	n = 9    # Number of points around each foot
	r = 0.07 # Radius of the height sensors around each toe.
	P = np.empty([n, 2])
	phi = 2*np.pi/n

	for i in range(n):
		angle_i = alpha + i* phi
		P[i] = np.array([x + r * np.cos(angle_i), 
						y +  r * np.sin(angle_i)])
	
	return P

class simulation:
	""" Simulation. """
	def __init__(self,
				terrain_file: str, 
				giadog_file: str,
				bullet_server,
				mesh_scale: List[float]=[1/50, 1/50, 1],
				actuated_joints_ids: List[int] = [7,8,9, 11,12,13, 16,17,18, 20,21,22], 
				
				thighs_ids: List[int]  = [8,12,17,18],
				shank_ids: List[int]  = [9,13, 18, 22],
				toes_ids: List[int] =  [10, 14, 19, 23],
				): 
		"""
		[TODO:DESCRIPTION]
		"""
		
		self.giadog_urdf_file = giadog_file
		self.p = bullet_server
		self.mesh_scale = mesh_scale
		self.terrain_file = terrain_file
		
		# This array is used to calculate the robot toes heightfields 
		self.terrain_array    = np.loadtxt(open(self.terrain_file))
		x_size, y_size = self.terrain_array.shape 
		x_space = np.linspace(0, x_size*mesh_scale[0], num = x_size)
		y_space = np.linspace(0, y_size*mesh_scale[1], num = y_size)


		# Function that given an x, y coordinate returns the height of that point
		self.get_terrain_height = interp2d(x_space, y_space, 
										self.terrain_array, kind='linear')
		
		self.actuated_joints_ids = actuated_joints_ids
		self.thighs_ids = thighs_ids
		self.shank_ids = shank_ids
		self.toes_ids = toes_ids
	
	def initialize(self,
					x_o:float = 0.0, 
					y_o:float = 0.0,
					):
		"""
		
		Initializes a pybullet simulation, setting up the terrain, gravity and the 
		quadruped in the pybullet enviroment. (And enabiling the torque sensors in the 
		quadruped foots/toes)

		Reference:

		Args:
			x: float  -> x coordintate of the robot initial position (In the world frame)
			y: float  -> y coordintate of the robot initial position (In the world frame)
		Return:
			None
		"""
		self.p.connect(self.p.GUI)
		terrain_shape = self.p.createCollisionShape(
			shapeType = self.p.GEOM_HEIGHTFIELD, 
			meshScale = self.mesh_scale,
			fileName  = self.terrain_file, 
			heightfieldTextureScaling=128
		)
		self.terrain = p.createMultiBody(0, terrain_shape)

		self.p.resetBasePositionAndOrientation(self.terrain,[0,0,0], [0,0,0,1])
		self.p.setGravity(0, 0, -9.807)

		self.gravity_vector = np.array([0, 0, -9.807])


		# 
		self.quadruped = self.p.loadURDF(self.giadog_file, 
									  [x_o, y_o, self.get_terrain_height(x_o, y_o) + 0.3])

		#Torque sensors are enable on the quadruped toes

		for toe_id in self.toes_ids:
			self.p.enableJointForceTorqueSensor(
				bodyUniqueId = self.quadruped,
				jointIndex = toe_id,
				enableSensor = True,
			)
	


	def update_sensor_output(self):
		"""
		
		Updates the sensor states for the current simulation steps.

		It updates: [TODO]

		Reference:

		Args:
			self: simulation  ->  Simulation class
		Return:
			None
		"""

		#-----------------------------Base Velocity--------------------------------------#

		self.base_linear_velocity, self.base_angular_velocity  = np.array(
												self.p.getBaseVelocity(self.quadruped)
												)

		#-----------------------------Contact info---------------------------------------#
		
		# Toes//Foot//Patitas
		self.terrain_normal_at_each_toe = np.zeros([4, 3])
		self.normal_force_at_each_toe = np.zeros([4])
		self.toes_contact_states = np.zeros([4])
		for i, toe_id in enumerate(self.toes_ids):

			# Privileged information
			toe_contact_info = self.p.getContactPoints(
												   bodyA  = self.quadruped, 
												   bodyB = self.terrain, 
												   linkIndexA = toe_id)
			if toe_contact_info == (): #No contact case
				
				self.terrain_normal_at_each_toe[i] = (0,0,0)
				self.normal_force_at_each_toe[i] = 0 
				self.toes_contact_states[i] = 0
			
			else:
				toe_contact_info = ContactInfo(*toe_contact_info)
				self.terrain_normal_at_each_toe[i] = toe_contact_info.contactNormalOnB 
				self.normal_force_at_each_toe[i] = toe_contact_info.normalForce
				self.toes_contact_states[i] = 1
		
		# Thighs//Muslos
		self.thighs_contact_states = np.zeros([4])
		for i, thigh_id in enumerate(self.thighs_ids):

			thigh_contact_info = self.p.getContactPoints(
												   bodyA  = self.quadruped, 
												   bodyB = self.terrain, 
												   linkIndexA = thigh_id)
			
			self.thighs_contact_states[i] = float(thigh_contact_info != ())
			
		# Shanks//canillas
		self.shanks_contact_states = np.zeros([4])
		for i, shank_id in enumerate(self.shanks_ids):

			shank_contact_info = self.p.getContactPoints(
												   bodyA  = self.quadruped, 
												   bodyB = self.terrain, 
												   linkIndexA = shank_id)

			self.shanks_contact_states[i] = float(shank_contact_info != ())

		
		#--------------------------------------------------------------------------------#
		
		#-----------------------------Height Scan----------------------------------------# 
		self.height_scan_at_each_toe = np.zeros([4, 9]) # 9 scan points around each toe
		for i, toe_link_state in enumerate(
									self.p.getLinkStates(self.quadruped, self.toes_ids)
									):
			
			toe_link_state =  LinkState(*toe_link_state)
			
			toe_orientation = toe_link_state.linkWorldOrientation
			toe_position =  toe_link_state.linkWorldPosition
		
			# Height scan around each foot 
			roll, pitch, yaw = self.p.getEulerFromQuaternion(toe_orientation)
			x,y =  toe_position[0], toe_position[1] 
			P = get_foot_height_scan_coordinates(x,y,yaw) 
			self.height_scan_at_each_toe[i] = [self.get_terrain_height(x, y) 
											   for (x,y) in P]
		#--------------------------------------------------------------------------------#


		#-----------------------------Toe Force Sensors----------------------------------#
		toe_force_sensor_threshold = 6 # Newtons 
		self.toe_force_sensor = np.zeros(4) # 4 = Number of toes
		for i, toe_joint_state in enumerate(self.p.getJointStates( 
											bodyUniqueId = self.quadruped,
											jointIndex = self.toes_ids)):
			
			toe_joint_state = JointState(*toe_joint_state) 
			# "Analog" toe force sensor
			F_x, F_y, F_z, M_x, M_y, M_z = toe_joint_state.jointReactionForces
			self.toe_force_sensor[i] =  float((abs(F_x) + abs(F_y) + abs(F_z) ) 
												> toe_force_sensor_threshold)  

		#--------------------------------------------------------------------------------#	


		#--------------------------Actuated joints sensors-------------------------------#
		# (Position/Velocity/Torque)	
		# Joint angles
		self.joint_angles = np.zeros(12) # 12 = Number of DOF // Controlled joints
		self.joint_velocities = np.zeros(12)
		self.joint_torques = np.zeros(12) # For reward calculations   
		for i, j_state in enumerate(self.p.getJointStates(
												bodyUniqueId = self.quadruped,
												jointIndex = self.actuated_joints_ids)):

			j_state = JointState(*j_state)

			self.joint_angles[i] = j_state.jointPosition
			
			self.joint_velocities[i] = j_state.jointVelocity

			self.joint_torques[i] = j_state.appliedJointMotorTorque
		
		#--------------------------------------------------------------------------------#
		




		


	def test_terrain(
		self, 
		terrain_file: str, 
		giadog_file: str,
		mesh_scale: List[int]=[1/50, 1/50, 1],
		init: Tuple[int, int]=(0, 0)
	):
		"""
			ESP:
			
			Genera una simulacion de prueba para un terreno especificado.

			Args:
				terrain_file: str  ->  Direccion del archivo que contiene el terreno.
				giadog_file: str  ->  Direccion del archivo que contiene la especificacion
					del agente.
				mesh_scale: List[int]  ->  Lista 3D que representa la scala del terreno.
				init: Optional[Tuple[int, int]]  ->  Coordenada inicial. La altura es 
					calculada de forma automatica. Valor por defecto: (0, 0).

			ENG:
		"""
		# Obtenemos el numero de filas y columnas
		with open(terrain_file, 'r') as f:
			lines = f.readlines()
			rows = len(lines)
			cols = len(lines[0].split())

			pos_x = rows // 2 + int(init[0] / mesh_scale[0])
			pos_y = cols // 2 + int(init[1] / mesh_scale[1])
			h = float(lines[pos_x].split()[pos_y][:-1]) / (2 * mesh_scale[2])

		# Conectamos pybullet
		p.connect(p.GUI)
		#p.setAdditionalSearchPath(pd.getDataPath())
		
		# Configuracion del terreno
		terrain_shape = p.createCollisionShape(
			shapeType=p.GEOM_HEIGHTFIELD, 
			meshScale=mesh_scale,
			fileName=terrain_file, 
			heightfieldTextureScaling=128
		)
		terrain = p.createMultiBody(0, terrain_shape)
		p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
		p.setGravity(0, 0, -9.807)

		# Agregamos al agente GIAdog
		p.loadURDF(giadog_file, [init[0], init[1], h])

		while True: 
			p.stepSimulation()
			time.sleep(0.01)
	
	def test_sensors(self):
		"""
			ESP:
			
			Genera una simulacion para probar de los "sensores" del robot.

			Args:
				terrain_file: str  ->  Direccion del archivo que contiene el terreno.
				giadog_file: str  ->  Direccion del archivo que contiene la especificacion
					del agente.
				mesh_scale: List[int]  ->  Lista 3D que representa la scala del terreno.
				init: Optional[Tuple[int, int]]  ->  Coordenada inicial. La altura es 
					calculada de forma automatica. Valor por defecto: (0, 0).

			ENG:
		"""

		
		while True: 
			self.p.stepSimulation()
			time.sleep(1/240)


		

if __name__ == '__main__':
	def syntax_error():
		print(
			"Invalid syntax. Use:\n\n" +\
			"  \033[1mpython simulation.py --test\033[0m \033[3;4mTERRAIN\033[0m " +\
				"\033[3;4mGIADOG\033[0m \033[3;4mROW\033[0m \033[3;4mCOL\033[0m\n",
			file=stderr
		)
		exit(1)

	if len(argv) < 6: syntax_error()

	if argv[1] == "--test":
		simulation().test_terrain(argv[2], argv[3], init=(float(argv[4]), float(argv[5])))