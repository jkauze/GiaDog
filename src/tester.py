"""
Date    : 11/08/2021
Author  : Eduardo López
Project : Graduation Thesis: GIAdog

Purpose: The purpose of this script is to render the terrain represented by a .txt file to check if it is correct

Note: Testing script

Part of the code was taken from:
"""
import pybullet as p
import pybullet_data as pd
import time
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())




terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, meshScale=[1, 1, 1],fileName = "map.txt", heightfieldTextureScaling=128)
terrain  = p.createMultiBody(0, terrainShape)
p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
p.loadURDF("mini_ros/urdf/spot.urdf", [0,0,0])
p.setGravity(0, 0, -9.807)


while True:
	p.stepSimulation()
	time.sleep(0.01)
	