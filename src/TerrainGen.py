from typing import *
from sys import argv, stderr
from perlin_noise import PerlinNoise

import matplotlib.pyplot as plt
import numpy as np

class TerrainGen:
  """
    Generador de Terrenos.
  """
  def __init__(self, filename: str):
    self.instructions = {
      'TERRAIN': self.terrain,
      'PERLIN': self.perlin,
      'STEP': self.step,
      'HILL': self.hill,
      'STAIR': self.stair,
      'MAINCRA': self.maincra,
      'SAVE': self.save,
      'PLOT': self.plot
    }
    self.terrain_ = [[0]]
    self.read_instructions(filename)

  def error(self, msg: str):
    """ Imprime un mensaje de error. """
    print(f'\033[1;31mError.\033[0m {msg}', file=stderr)
    exit(1)

  def terrain(self, rows: int, cols: int): 
    """
      Genera el terreno inicial.

      Args:
        rows: int ->  Numero de filas del mapa.
        cols: int ->  Numero de columnas del mapa.
        smooth: float ->  Suavidad del terreno usando Perlin Noise. Si su valor es 0, no
          se aplicara el ruido.
        seed: int ->  Semilla para la aleatoriedad del Perlin Noise.
    """
    # Casteamos los argumentos pues seran strings al ser llamados desde el interpretador.
    self.rows = int(rows)
    self.cols = int(cols)
    self.terrain_ = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

  def perlin(self, max_h: float, smooth: float, seed: int):
    """
      Agrega el Perlin Noise al terreno.

      Args:
        max_h: float  ->  Maxima altura de las colinas.
        smooth: float ->  Suavidad del terreno usando Perlin Noise. Si su valor es 0, no
          se aplicara el ruido. Debe ser mayor a 0.
        seed: int ->  Semilla para la aleatoriedad del Perlin Noise.
    """
    # Casteamos los argumentos pues seran strings al ser llamados desde el interpretador.
    max_h = float(max_h)
    smooth = float(smooth)
    seed = int(seed)

    noise = PerlinNoise(octaves=smooth, seed=seed)
    terrain = [[noise([i/self.rows, j/self.cols]) for j in range(self.cols)] for i in range(self.rows)]

    # Escalamos los valores del terreno al rango [0, MAX)
    min_v = min([min(r) for r in terrain])
    max_v = max([max(r) for r in terrain]) - min_v
    terrain = [[(terrain[i][j] - min_v) * max_h / max_v for j in range(self.cols)] for i in range(self.rows)]

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
    # Casteamos los argumentos pues seran strings al ser llamados desde el interpretador.
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
        c: float  ->  Curvatura. Si es 0, la colina parecera un cono. Mientras tienda
          a infinito, la colina tendera a un cilindro.
        rough: float  ->  Indice de rugosidad.  
    """
    # Casteamos los argumentos pues seran strings al ser llamados desde el interpretador.
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
        row: int  ->  Fila donde se encuentra la parte superior izquierda de la linea que
          inicia la escalera.
        col: int  ->  Columna donde se encuentra la parte superior izquierda de la linea que
          inicia la escalera.
        o: str  ->  Orientacion de la escalera. Puede ser N (Norte), E (Este), S (Sur) o W 
          (Oeste).
        w: int  ->  Ancho de cada escalon.
        l: int  ->  Largo de cada escalon.
        h: int  ->  Altura de cada escalon.
        n: int  ->  Numero de escalones.
    """
    # Casteamos los argumentos pues seran strings al ser llamados desde el interpretador.
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

  def maincra(self, w: int, smooth: float, seed: int):
    """ 
      Agrega un terreno de solo cubos. 
    
      Args:
        w: int  ->  Anchura de cada cubo.
    """
    w = int(w)
    smooth = float(smooth)
    seed = int(seed)

    noise = PerlinNoise(octaves=smooth, seed=seed)
    row = 0
    while row < self.rows:
      col = 0
      while col < self.cols:
        self.step(row, col, w, w, noise([row/self.rows, col/self.cols]))
        col += w
      row += w

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
          self.error(f'Line {i+1}. Undefined instruction \033[1m{instr}\033[0m.')
        self.instructions[instr](*line)

tg = TerrainGen(argv[1])