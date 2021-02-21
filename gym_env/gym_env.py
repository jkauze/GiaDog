import gym
import pybullet as p
from bullet_dataclasses import Joint, JointState, sensors_state
import numpy as np

import pybullet_data






FPS = 50



def sensor_output(body, acctuated_joint_ids = [], toe_joint_ids = [], toe_force_threshold = 10, output_type = "list"):
	"""
	body: int : The unique id of the robot body 
	acctuated_joint_ids: [int] : The indexes of the joints corresponding to the joints of the robot which have servos in them. (Acctuated joints)
	toe_joint_ids: [int]: The indezes of the robot toes (pies). We want to know wether they are touching the ground or not.
	toe_force_threshold: float :The force treshold that indicathes whether a toe is touching the ground or not.
	output_type: str : Defines the type of the output  must be either 'list' or 'dataclass'. Dataclass can be used for descriptive tasks (Real time evaluation), list can be used to train the agents.
	"""

	position, orientation = p.getBasePositionAndOrientation(body) 
	linear_velocity, angular_velocity = p.getBaseVelocity(body)
	
	roll, pitch, yaw = p.getEulerFromQuaternion(orientation) #Datos del magnetometro
	ω_x, ω_y, ω_z = angular_velocity #Datos del giroscopio

	toe_states = []
	for i, toe_id in zip(range(len(toe_joint_ids)) ,toe_joint_ids):
		
		F_x, F_y, F_z, M_x, M_y, M_z = JointState(*p.getJointState(bodyUniqueId = body,
													  jointIndex = toe_id)).jointReactionForces
		
		toe_states.append( float((abs(F_x) + abs(F_y) + abs(F_z) ) > toe_force_threshold) ) 


	joint_angles = [] 
		
	for joint_id in acctuated_joint_ids:
		
		joint_angles.append(JointState(*p.getJointState(bodyUniqueId = body,
										  jointIndex = joint_id)).jointPosition)

	# sensors_state([roll, pitch, yaw],[ω_x, ω_y, ω_z], joint_angles, toe_states) # 

	if output_type == "dataclass":

		return sensors_state([roll, pitch, yaw],[ω_x, ω_y, ω_z], joint_angles, toe_states)
	
	elif output_type == "list":

		return [roll, pitch, yaw, ω_x, ω_y, ω_z, *joint_angles, *toe_states]
	
	else:

		raise "Invalid Argument for output_type kwarg; must be either 'list' or  'dataclass'"



class Behaviour():

	def __init__(self, settings):
		"""
		"""
		self.name = settings['name'] # str

		self.goal_twist = np.array(settings['goal_twist']) # [ω_x, ω_y, ω_z, v_x, v_y, v_z] list

		self.min_z = settings['min_z'] # Minimal z value we ddont 


	def compute_reward(self, body):
		"""
		"""
		
		position, orientation = p.getBasePositionAndOrientation(body) 
		
		linear_velocity, angular_velocity = p.getBaseVelocity(body)

		x, y, z = position

		err =  sum(abs(self.goal_twist - np.array([*angular_velocity, *linear_velocity]))) 

		
			
		return [err, z < self.min_z, 0] # reward and done condition (If the robot falls a certain trehsold)
		






class Quadruped_Control(gym.Env):
	"""docstring for Quadruped_Control"""

	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : FPS
	}

	continuous = False
	

	def __init__(self, settings):
		"""
		settings: dict
		"""
		super(Quadruped_Control, self).__init__()

		self.spot_path           = settings['spot_path'] # path of the .urdf of spot (referenced from the pybullet_data.getDataPath() path)
		self.actuated_joints_ids = settings['actuated_joints_ids']#    
		self.toes_ids            = settings['toes_ids'] # 
		self.terrain_path        = settings['terrain_path'] #path of the .urdf file of the terrain (from the pybullet_data.getDataPath() path  usually "/usr/local/lib/python3.6/dist-packages/pybullet_data")
		self.initial_position    = settings['initial_position']#[x,y,z]

		self.toe_force_sensor_threshold = settings['toe_force_sensor_threshold'] # 10 is a good value

		

		if settings['control_mode'] == "position":
			self.control_mode = p.POSITION_CONTROL

		elif settings['control_mode'] == "velocity":
			self.control_mode = p.VELOCITY_CONTROL

		
		self.control_mode_label = settings['control_mode'] 


		self.behaviour = settings['behaviour'] # Indicates the behaviour to be evaluated by the enviroment: ej: standstill, rotate, move foward, move sideways to the left, etc. 

		self.camera_settings = settings['camera_settings'] 

		self.video = []

		self.record =  settings['record'] # booelan: if True, each step of the simulation an image would be redered and appended to self.video according to settings['camera_settings']. Thi is mainly for colab/jupyter use.

		
		self.action_space = np.zeros( (len(self.actuated_joints_ids),1))

		self.observation_space =  np.zeros( (6 + len(self.actuated_joints_ids) + len(self.toes_ids),1))






	def reset(self):
		"""
		Resets the enviroment:
		Reload the urdf files of the robot and the terrain

		"""

		# We reset the simulation with pybullet 

		p.resetSimulation()


		# We load the terrain urdf file
		
		p.loadURDF(self.terrain_path)

		# We load the spot urdf file
		
		self.spot = p.loadURDF(self.spot_path, self.initial_position)


		# Set the gravity to *9.807 m/s^2
		p.setGravity(0, 0, -9.807)


		# We get the acctuated joints force limits

		self.actuated_force_limits = []

		for joint_id in self.actuated_joints_ids:
			joint = Joint(*p.getJointInfo(self.spot, joint_id))
			self.actuated_force_limits.append(joint.maxForce) 


		# We activate force sensors in spots toes

		for toe_id in self.toes_ids  :
			p.enableJointForceTorqueSensor(
				bodyUniqueId = self.spot ,
				jointIndex = toe_id,
				enableSensor = True,
			)




		self.sensor_state =  sensor_output(self.spot, acctuated_joint_ids = self.actuated_joints_ids, toe_joint_ids = self.toes_ids, toe_force_threshold = self.toe_force_sensor_threshold, output_type = "list")

		return self.sensor_state

	def record_video(self):
		"""

		The function appends an image, according to the self.camera_settings parameters, into self.video using pybullets getCameraImage function.
		
		Note1: 
		To render the video in google colaboratory you can use the following lines of code:
		
		from numpngw import write_apng
		from IPython.display import Image 
		
		write_apng('video.png', self.video, delay=20)
		Image(filename='video.png')


		Note2: For this fucntion to be fast check PyBullet.isNumpyEnabled() (must be True, if not check PyBullet.getCameraImage in the pybullet documentation)
		"""

		if self.camera_settings["cameraTargetPosition"] == "spot":
			
			position, __ = p.getBasePositionAndOrientation(self.spot) 
		
		else:
			position =  self.camera_settings["cameraTargetPosition"]

		
		width  = self.camera_settings['width']
		height = self.camera_settings['height']
		
		img_arr = p.getCameraImage(
			width,
			height,
			viewMatrix=p.computeViewMatrixFromYawPitchRoll(
				
				cameraTargetPosition = position,
				
				distance = self.camera_settings['distance'],
				yaw      = self.camera_settings['yaw'],
				pitch    = self.camera_settings['pitch'],
				roll     = self.camera_settings['roll'],
				
				upAxisIndex=self.camera_settings['upAxisIndex'], #(2 should be ok)
			),
			projectionMatrix=p.computeProjectionMatrixFOV(
				
				fov     = self.camera_settings['fov'],
				aspect  = width/height,
				nearVal = self.camera_settings['nearVal'],# 0.01 is ok
				farVal  = self.camera_settings['farVal'], # 100 is ok
			
			),
			shadow = self.camera_settings['shadow'], # [1,1,1] is ok
			lightDirection=self.camera_settings['lightDirection'], # [1,1,1] is ok
		)
		width, height, rgba, depth, mask = img_arr

		self.video.append(rgba)


	def step(self, action):
		"""
		action: [float]
		"""
		if self.control_mode_label == "velocity":

			p.setJointMotorControlArray(
			bodyUniqueId = self.spot, 
			jointIndices = self.actuated_joints_ids,
			controlMode = self.control_mode,
			targetVelocities = action,
			forces = self.actuated_force_limits,
			)

		elif self.control_mode_label == "position":

			p.setJointMotorControlArray(
			bodyUniqueId = self.spot, 
			jointIndices = self.actuated_joints_ids,
			controlMode = self.control_mode,
			targetPositions = action,
			forces = self.actuated_force_limits,
			)

		p.stepSimulation()

		self.sensor_state =  sensor_output(self.spot, acctuated_joint_ids = self.actuated_joints_ids, toe_joint_ids = self.toes_ids, toe_force_threshold = self.toe_force_sensor_threshold, output_type = "list")

		if self.record:
			self.record_video()

		reward, done, _ = self.behaviour.compute_reward(self.spot) 


		return self.sensor_state, reward, done, _


