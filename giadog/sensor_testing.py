"""
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/08/13

	Script to test the pybullet simulations sensors outputs.
"""

import pybullet as p
import pybullet_data as pd
from giadog.src.simulation.bullet_dataclasses import *
import time
import numpy as np





	




def sensor_output(robot, actuated_joints_ids, toes_ids, terrain, output_type = "dataclass"):
	"""

	"""
	
	toe_force_sensor_threshold = 3.0


	position, orientation =p.getBasePositionAndOrientation(robot) 
	linear_velocity, angular_velocity =p.getBaseVelocity(robot)

	roll, pitch, yaw =p.getEulerFromQuaternion(orientation) #Datos del magnetometro
	ω_x, ω_y, ω_z = angular_velocity #Datos del giroscopio

	toe_states = []
	for i, toe_id in zip(range(len(toes_ids)) ,toes_ids):
		
		F_x, F_y, F_z, M_x, M_y, M_z = JointState(*p.getJointState( bodyUniqueId = robot,
																	jointIndex = toe_id)
																	).jointReactionForces

		
		toe_states.append( float((abs(F_x) + abs(F_y) + abs(F_z) ) 
							> toe_force_sensor_threshold) ) 


	joint_angles = [] 
		
	for joint_id in actuated_joints_ids:
		
		joint_angles.append(JointState(*p.getJointState(bodyUniqueId = robot,
											jointIndex = joint_id)).jointPosition)





	#Privileged information.
	contact_info = []
	for toe_id in toes_ids:
		contact_info.append(*p.getContactPoints(bodyA = robot, 
											   bodyB = terrain, 
											   linkIndexA = toe_id))
	# Terrain normal at each foot 12 X
	for info in contact_info:
		print("Normal Force")
		print(info)
		print("Normal Vector")
		#print(info.contactNormalOnB)
		print("")

	# Height scan around each foot 36 X
	# Foot contact forces 4 X
	# Foot contact states 4 X
	# Thigh contact states 4 X
	# Shank contact states 4 X
	# Foot-ground friction coefficients 4 X
	# External force applied to the base 


	if output_type == "dataclass":

		return SensorsState([roll, pitch, yaw],[ω_x, ω_y, ω_z], joint_angles, toe_states)

	elif output_type == "list":

		return [roll, pitch, yaw, ω_x, ω_y, ω_z, *joint_angles, *toe_states]

	else:

		raise "Invalid Argument for output_type kwarg; must be either 'list' or  'dataclass'"




if __name__ == '__main__':

	giadog_file = "mini_ros/urdf/spot.urdf"
	p.connect(p.GUI)
	p.resetSimulation()
	p.setAdditionalSearchPath(pd.getDataPath())
	plane  = p.loadURDF('plane.urdf')
	blacky = p.loadURDF(giadog_file, [0, 0, 0.3])
	p.setGravity(0, 0, -9.807)
	dt = 1/240
	t = 0

	# Getting the urdf joint info
	print(f"blacky unique ID: {blacky}")
	for i in range(p.getNumJoints(blacky)):
		joint = Joint(*p.getJointInfo(blacky, i))
		print(joint)
	

	actuated_joints_ids = [7,8,9, 11,12,13, 16,17,18, 20,21,22] 
	toes_ids =  [10, 14, 19, 23]
	
	for toe_id in toes_ids:
		p.enableJointForceTorqueSensor(
			bodyUniqueId = 1,
			jointIndex = toe_id,
			enableSensor = True,
		)

	while True:
		p.stepSimulation()
		time.sleep(dt)
		t = t + dt
		if t%1 < 0.005:
			#print(sensor_output(blacky, actuated_joints_ids, toes_ids, plane))
			"""
			toe_contact_info = p.getContactPoints(
												   bodyA  = blacky, 
												   bodyB = plane, 
												   linkIndexA = 10)
			print(toe_contact_info)
			"""
			info_a, infor_b = np.array(p.getBaseVelocity(blacky))
			print(info_a)
			print(" ")
			print(type(info_a))
			print(" ")
