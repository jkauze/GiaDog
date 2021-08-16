"""
	Authors: Amin Arriaga, Eduardo Lopez
	Project: Graduation Thesis: GIAdog
	Last modification: 2021/08/13

	Generate a .tg file representing the height field of the terrain to train GIAdog 
	agents.
"""
# Utilidades
from typing import *
from sys import argv, stderr
from os import remove
import matplotlib.pyplot as plt
import numpy as np
import uuid
import time

# Generacion de terrenos
from perlin_noise import PerlinNoise

# Simulacion
import pybullet as pb
import pybullet_data as pd

class terrain_gen:
	""" Generador de Terrenos. """

	def __init__(self, filename: Optional[str]=None):
		self.instructions = {
			'TERRAIN': self.terrain,
			'PERLIN': self.perlin,
			'STEP': self.step,
			'HILL': self.hill,
			'STAIR': self.stair,
			'MAINCRA': self.adaptative_maincra,
			'SAVE': self.save,
			'PLOT': self.plot
		}
		self.terrain_ = [[0]]
		if filename != None: self.read_instructions(filename)

	def error(self, msg: str):
		""" Imprime un mensaje de error. """
		print(f'\033[1;31mError.\033[0m {msg}', file=stderr)
		exit(1)

	def terrain(self, rows: int, cols: int): 
		"""
			Genera el terreno inicial.

			Args:
				rows: int  ->  Numero de filas del mapa.
				cols: int  ->  Numero de columnas del mapa.
				smooth: float  ->  Suavidad del terreno usando Perlin Noise. Si su valor 
					es 0, no se aplicara el ruido.
				seed: int  ->  Semilla para la aleatoriedad del Perlin Noise.
		"""
		# Casteamos los parametros pues seran str al ser llamados desde el interpretador.
		self.rows = int(rows)
		self.cols = int(cols)
		self.terrain_ = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

	def perlin(self, max_h: float, smooth: float, seed: int):
		"""
			Agrega el Perlin Noise al terreno.

			Args:
				max_h: float  ->  Maxima altura de las colinas.
				smooth: float  ->  Suavidad del terreno usando Perlin Noise. Debe ser 
					mayor a 0.
				seed: int ->  Semilla para la aleatoriedad del Perlin Noise.
		"""
		# Casteamos los parametros pues seran str al ser llamados desde el interpretador.
		max_h = float(max_h)
		smooth = float(smooth)
		seed = int(seed)

		epsilon = 1e-10
		noise = PerlinNoise(octaves=smooth + epsilon, seed=seed)
		terrain = [
			[noise([i/self.rows, j/self.cols]) for j in range(self.cols)] 
			for i in range(self.rows)
		]

		# Escalamos los valores del terreno al rango [0, MAX)
		min_v = min([min(r) for r in terrain])
		max_v = max([max(r) for r in terrain]) - min_v
		terrain = [
			[(terrain[i][j] - min_v) * max_h / max_v for j in range(self.cols)] 
			for i in range(self.rows)
		]

		for i in range(self.rows):
			for j in range(self.cols):
				self.terrain_[i][j] += terrain[i][j]

	def step(self, row: int, col: int, w: int, l: int, h: float): 
		"""
			Agrega un 'cubo' al terreno.

			Args:
				row: int  ->  Fila donde se encuentra la esquina superior izquierda.
				col: int  ->  Columna donde se encuentra la esquina superior izquierda.
				w: int  ->  Ancho del cubo.
				l: int  ->  Largo del cubo.
				h: int  ->  Altura del cubo.
		"""
		# Casteamos los parametros pues seran str al ser llamados desde el interpretador.
		row = int(row)
		col = int(col)
		w = int(w)
		l = int(l)
		h = float(h)

		for i in range(row, min(row + w, self.rows)):
			for j in range(col, min(col + l, self.cols)):
				self.terrain_[i][j] += h

	def hill(self, row: int, col: int, r: float, h: float, c: float, rough: float):
		"""
			Agrega una colina redonda al terreno

			Args:
				row: int  ->  Fila donde se encuentra el centro.
				col: int  ->  Columna donde se encuentra el centro.
				r: float  ->  Radio
				h: float  ->  Altura.
				c: float  ->  Curvatura. Si es 0, la colina parecera un cono. Mientras 
					tienda a infinito, la colina tendera a un cilindro.
				rough: float  ->  Indice de rugosidad.  
		"""
		# Casteamos los parametros pues seran str al ser llamados desde el interpretador.
		row = int(row)
		col = int(col)
		r = int(r)
		c = int(c)
		h = float(h)
		rough = float(rough)

		for i in range(self.rows):
			for j in range(self.cols):
				d = (i - row) ** 2 + (j - col) ** 2
				if d <= r ** 2:
		  			self.terrain_[i][j] += ((1 - d / (r ** 2)) * h) ** (1 / (1 + c))

	def stair(self, row: int, col: int, o: str, w: int, l: int, h: float, n: int):
		"""
			Agrega una escalera al terreno.

			Args:
				row: int  ->  Fila donde se encuentra la parte superior izquierda de la 
					linea que inicia la escalera.
				col: int  ->  Columna donde se encuentra la parte superior izquierda de 
					la linea que inicia la escalera.
				o: str  ->  Orientacion de la escalera. Puede ser N (Norte), E (Este), S 
					(Sur) o W (Oeste).
				w: int  ->  Ancho de cada escalon.
				l: int  ->  Largo de cada escalon.
				h: int  ->  Altura de cada escalon.
				n: int  ->  Numero de escalones.
		"""
		# Casteamos los parametros pues seran str al ser llamados desde el interpretador.
		row = int(row)
		col = int(col)
		w = int(w)
		l = int(l)
		h = float(h)
		n = int(n)

		if o == 'E':
			for i in range(n): self.step(row, col + i*l, w, l, i*h)
		elif o == 'S':
			for i in range(n): self.step(row + i*w, col, w, l, i*h)
		elif o == 'W':
			for i in range(n): self.step(row, col - (i+1)*l, w, l, i*h)
		elif o == 'N':
			for i in range(n): self.step(row - (i+1)*w, col + i*w, w, l, i*h)
		else:
		  	self.error(f'Undefined orientation {o}.')

	def plot(self):
		""" Graficamos el terreno actual. """
		plt.imshow(self.terrain_, cmap='gray')
		plt.show()

	def save(self, filename: str):
		""" Guardamos el terreno en un archivo txt. """
		np.savetxt(filename, np.array(self.terrain_), delimiter=",")

	def read_instructions(self, filename: str):
		""" Ejecutamos el interpretador. """
		with open(filename, 'r') as f:
			for i, line in enumerate(f.readlines()):
				line = line.split()
				# Las lineas que comiencen con '#' seran ignorados.
				if not line or line[0][0] == "#": continue
				# El primer token define la instruccion.
				instr = line.pop(0)
				# Si el primer token no corresponde a ninguna instruccion, error
				if not instr in self.instructions: 
					self.error(
						f'Line {i+1}. Undefined instruction \033[1m{instr}\033[0m.'
					)
				self.instructions[instr](*line)

	def adaptative_hills(self, r: float, f: float, a: float, seed: int):
		"""
			Generacion de terrenos de colinas adaptativos.

			Args:
			r: float  ->  Indice de aspereza del terreno.
			f: float  ->  Indice de frecuencia con la que aparecen colinas.
			a: float  ->  Altura maxima del terreno.
			seed: int ->  Semilla.
		"""
		if not 0 <= r <= 1: 
			error(f'Indice de aspereza {r} invalido. Debe estar en el rango [0,1].')
		if not 0 <= f <= 1: 
			error(f'Indice de frecuencia {f} invalido. Debe estar en el rango [0,1].')
		
		self.terrain(100, 100)

		# Maxima aspereza y frecuencia
		MAX_A = 0.1
		MAX_F = 7

		# Creamos las colinas
		self.perlin(a, f * MAX_F, seed)
		# Agregamos la aspereza
		self.perlin(r * MAX_A, 50, seed+1)
		self.perlin(r * MAX_A / 5, 70, seed+1)

	def adaptative_maincra(self, w: int, h: float, seed: int):
		"""
			Generacion de terrenos maincra adaptativos.

			Args:
				w: int  ->  Ancho de cada bloque
				h: float  ->  Altura maxima
		"""
		self.terrain(100, 100)

		noise = PerlinNoise(octaves=20, seed=seed)

		# Calculamos el minimo y maximo de las alturas usando Perlin Noise.
		terrain = [
			[noise([i/self.rows, j/self.cols]) for j in range(self.cols)] 
			for i in range(self.rows)
		]
		min_v = min([min(r) for r in terrain])
		max_v = max([max(r) for r in terrain]) - min_v

		row = 0
		while row < self.rows:
			col = 0
			while col < self.cols:
				h_k = (terrain[row // w][col // w] - min_v) * h / max_v
				self.step(row, col, w, w, h_k)
				col += w
			row += w

	def adaptative_stairs(self, w: float, h: float):
		"""
			Generacion de terrenos de escaleras adaptativos.

			Args:
				w: float  ->  Ancho de cada escalon.
				h: float  ->  Alto de cada escalon.
		"""
		self.terrain(500, 500)

		# Ancho de la zona inicial
		init_w = 50

		# Escaleras iniciales
		n0 = (100 - init_w - (100 - init_w) % w) // w
		self.stair(0, 0, 'E', 500, w, h, n0)

		# Zona inicial
		self.step(0, n0 * w, 500, init_w + (100 - init_w) % w, n0 * h)

		# Resto de escaleras subiendo
		n1 = (200 - init_w) // w
		self.step(0, 100, 500, 200 + 200 % w, (n0 + 1) * h)
		self.stair(0, 100, 'E', 500, w, h, n1)
		self.step(0, 100 + n1 * w, 500, init_w + (200 - init_w) % w + 200 % w, n1 * h)

		# Escaleras bajando
		n2 = 200 // w
		self.step(0, 300 + 200 % w, 500, n2 * w, (n0 + 1) * h + n1 * h - n2 * h)
		self.stair(0, 500, 'W', 500, w, h, n2)

	def simulation(self, mesh_scale: List[int]=[1/25,1/25,1]):
		"""
			Genera una simulacion de prueba con el terreno actual.

			Args:
				mesh_scale: List[int]  ->  Lista 3D que representa la scala del terreno.
		"""

		# Creamos un archivo temporal que almacenara el terreno
		filename = str(uuid.uuid4()) + '.temp.txt'
		self.save(filename)

		# Conectamos pybullet
		pb.connect(pb.GUI)
		pb.setAdditionalSearchPath(pd.getDataPath())
		
		# Configuracion del terreno
		terrain_shape = pb.createCollisionShape(
			shapeType=pb.GEOM_HEIGHTFIELD, 
			meshScale=mesh_scale,
			fileName=filename, 
			heightfieldTextureScaling=128
		)
		terrain = pb.createMultiBody(0, terrain_shape)
		pb.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])
		pb.setGravity(0, 0, -9.807)

		# Agregamos al agente GIAdog
		pb.loadURDF("mini_ros/urdf/spot.urdf", [0,0,1])

		# Eliminamos el archivo temporal
		remove(filename)

		while True: 
			pb.stepSimulation()
			time.sleep(0.01)
		

if __name__ == '__main__':
	tg = terrain_gen()
	tg.adaptative_stairs(int(argv[1]), float(argv[2]))
	tg.simulation()