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
		mesh_scale: List[int]=[1/50, 1/50, 1],
		init: Tuple[int, int]=(0, 0)
	):
		"""
			Genera una simulacion de prueba para un terreno especificado.

			Args:
				terrain_file: str  ->  Direccion del archivo que contiene el terreno.
				giadog_file: str  ->  Direccion del archivo que contiene la especificacion
					del agente.
				mesh_scale: List[int]  ->  Lista 3D que representa la scala del terreno.
				init: Optional[Tuple[int, int]]  ->  Coordenada inicial. La altura es 
					calculada de forma automatica. Valor por defecto: (0, 0).
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
		pb.loadURDF(giadog_file, [init[0], init[1], h])

		while True: 
			pb.stepSimulation()
			time.sleep(0.01)
		

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