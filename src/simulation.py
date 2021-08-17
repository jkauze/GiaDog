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
import pybullet as pb
import pybullet_data as pd

class simulation:
	""" Simulation. """
	def __init__(self): pass

	def test_terrain(
		self, 
		terrain_file: str, 
		giadog_file: str,
		mesh_scale: List[int]=[1/50,1/50,1]
	):
		"""
			Genera una simulacion de prueba para un terreno especificado.

			Args:
				terrain_file: str  ->  Direccion del archivo que contiene el terreno.
				giadog_file str  ->  Direccion del archivo que contiene la especificacion
					del agente.
				mesh_scale: List[int]  ->  Lista 3D que representa la scala del terreno.
		"""
		# Conectamos pybullet
		pb.connect(pb.GUI)
		pb.setAdditionalSearchPath(pd.getDataPath())
		
		# Configuracion del terreno
		terrain_shape = pb.createCollisionShape(
			shapeType=pb.GEOM_HEIGHTFIELD, 
			meshScale=mesh_scale,
			fileName=terrain_file, 
			heightfieldTextureScaling=128
		)
		terrain = pb.createMultiBody(0, terrain_shape)
		pb.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
		pb.setGravity(0, 0, -9.807)

		# Agregamos al agente GIAdog
		pb.loadURDF(giadog_file, [0,0,1])

		while True: 
			pb.stepSimulation()
			time.sleep(0.01)
		

if __name__ == '__main__':
	simulation().test_terrain(argv[1], argv[2])