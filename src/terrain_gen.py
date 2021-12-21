"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""

import numpy as np
from typing import *
from random import uniform
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise

class terrain_gen:
    """ Clase que permite la generación de terrenos para el entorno de simulación. """
    STEPS_FREQUENCY   = 10
    STEPS_NOISE       = 0.05
    ZONE_STAIRS_WIDTH = 25

    @staticmethod
    def terrain(rows: int, cols: int) -> np.ndarray:
        """
            Genera un nuevo terreno.

            Parametros:
            -----------
            rows: int
                Número de filas.
            cols: int
                Número de columnas.

            Return:
            -------
            numpy.ndarray, shape (M, N)
                Terreno con puros ceros de dimensiones rows x cols.
        """
        return np.zeros((rows, cols))

    @staticmethod
    def perlin(terrain: np.ndarray, height: float, octaves: float, seed: int):
        """
            Aplica el ruido de Perlin sobre un terreno.

            Parametros:
            -----------
            terrain: numpy.ndarray, shape (M, N)
                Terreno a modificar.
            height: float
                Maxima altura del ruido
            octaves: float
                Número de sub rectángulos en cada rango [0, 1].
            seed: int 
                Semilla específica con la que desea inicializar el generador aleatorio.
        """
        p_noise = PerlinNoise(octaves=octaves, seed=seed)
        rows, cols = terrain.shape

        # Calculamos el ruido.
        noise = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                noise[i][j] = p_noise([i/rows, j/cols])

        terrain += height * (noise - np.min(noise))

    @staticmethod
    def step(
            terrain: np.ndarray, 
            row: int, 
            col: int, 
            width: int, 
            lenght: int, 
            height: float
        ):
        """
            Agrega un cubo al terreno.

            Parametros:
            -----------
            terrain: numpy.ndarray, shape (M, N)
                Terreno a modificar.
            row: int
                Fila en la que se encuentra la esquina superior izquierda del cubo.
            col: int
                Columna en la que se encuentra la esquina superior izquierda del cubo.
            width: int
                Ancho (numero de filas) que ocupa el cubo.
            lenght: int
                Largo (numero de columnas) que ocupa el cubo.
            height: float
                Altura del cubo.
        """
        rows, cols = terrain.shape
        for i in range(row, min(rows, row + width)):
            for j in range(col, min(cols, col + lenght)):
                terrain[i][j] += height

    @classmethod
    def stair(
            cls,
            terrain: np.ndarray, 
            row: int, 
            col: int, 
            orientation: str, 
            width: int,
            length: int,
            height: float,
            n: int
        ):
        """
            Coloca una escalera en el terreno.

            Parametros:
            -----------
            terrain: numpy.ndarray, shape (M, N)
                Terreno a modificar.
            row: int
                Fila superior donde se encuentra la esquina del escalon mas bajo.
            col: int
                Columna superior donde se encuentra la esquina del escalon mas bajo.
            orientation: str, {'E', 'S', 'W', 'N'}
                Orientacion de la escalera.
            width: int
                Ancho de los escalones.
            length: int
                Largo de los escalones.
            height: float
                Altura de los escalones.
            n: int
                Numero de escalones.
        """
        if orientation == 'E':
            for i in range(n): 
                cls.step(terrain, row, col + i * length, width, length, i * height)
        elif orientation == 'S':
            for i in range(n): 
                cls.step(terrain, row + i * width, col, width, length, i * height)
        elif orientation == 'W':
            for i in range(n):
                cls.step(terrain, row, col - (i + 1) * length, width, length, i * height)
        elif orientation == 'N':
            for i in range(n):
                cls.step(terrain, row - (i + 1)* width, col, width, length, i * height)
        else:
            raise Exception(f'Unexpected orientation "\033[1;3m{orientation}\033[0m"')

    @staticmethod
    def goal(terrain: np.ndarray, row: int, col: int, height: float):
        """
            Coloca una vara vertical en el terreno.

            Argumentos:
            -----------
            terrain: np.ndarray, shape (N, N)  
                Terreno a modificar.
            row: int
                Fila donde se encuentra la vara.
            col: int
                Columna donde se encuentra la vara.
            heihgt: int
                Altura de la vara.
        """
        points = {
            (row-2, col), (row-1, col), (row, col), (row+1, col), (row+2, col),
            (row, col-2), (row, col-1), (row, col+1), (row, col+2),
            (row-1, col-1), (row-1, col+1), (row+1, col-1), (row+1, col+1)
        }

        rows, cols = terrain.shape
        base = max(terrain[x][y] for x, y in points if 0 <= x < rows and 0 <= y < cols)
        for x, y in points:
            if 0 <= x < rows and 0 <= y < cols:
                terrain[x][y] = base + height

    @classmethod
    def set_goal(cls, terrain: np.ndarray, height: float) -> Tuple[int, int]:
        """
            Coloca una vara vertical de forma aleatoria en algun punto de la 
            circunsferencia inscrita en el terreno
        """
        rows, cols = terrain.shape
        radio = min(rows, cols) // 2 - 2

        # Escogemos un angulo aleatorio
        angle = uniform(0, 2 * np.pi)
        row = int(radio * (1 + np.cos(angle)))
        col = int(radio * (1 + np.sin(angle)))
        cls.goal(terrain, row, col, height)

        return (row, col)

    @classmethod
    def hills(
            cls, 
            rows: int,
            cols: int,
            roughness: float,
            frequency: float, 
            amplitude: float, 
            seed: int
        ) -> np.ndarray:
        """
            Genera un terreno de colinas rugosas. Si el terreno es 500 x 500 y la escala
            es de 1:50, entonces un buen estimado de maxima dificultad para este tipo de
            terreno seria con los argumentos 
                * roughness = 0.05
                * frequency = 3.0
                * amplitude = 3.0

            Parametros:
            -----------
            rows: int
                Numero de filas del terreno.
            cols: int
                Numero de columnas del terreno.
            roughness: float
                Rugosidad del terreno. Debe estar preferiblemente en el rango [0, 0.05].
            frequency: float
                Frecuencia con la que aparecen las colinas. Debe ser positivo, 
                preferiblemente en el rango [0.2, 1]
            amplitude: float
                Altura maxima de las colinas.
            seed: int 
                semilla específica con la que desea inicializar el generador aleatorio.

            Return:
            -------
            np.ndarray
                Terreno resultante.
        """
        # Generamos el terreno
        terrain = cls.terrain(rows, cols)
        cls.perlin(terrain, amplitude, frequency, seed)

        # Agregamos la rugosidad
        for i in range(rows):
            for j in range(cols):
                terrain[i][j] += uniform(-roughness, roughness)

        return terrain

    @classmethod
    def steps(
            cls, 
            rows: int, 
            cols: int, 
            width: int, 
            height: float, 
            seed: int
        ) -> np.ndarray:
        """
            Genera un terreno de cubos. Si el terreno es 500 x 500 y la escala es de 
            1:50, entonces un buen estimado de maxima dificultad para este tipo de
            terreno seria con los argumentos 
                * width  = 10 (minimo)
                * height = 0.4

            Parametros:
            -----------
            rows: int
                Numero de filas del terreno.
            cols: int
                Numero de columnas del terreno.
            width: int
                Ancho y largo de los cubos.
            height: float
                Altura maxima de los cubos
            seed: int 
                semilla específica con la que desea inicializar el generador aleatorio.

            Return:
            -------
            np.ndarray
                Terreno resultante.
        """
        # Generamos el terreno
        terrain = cls.terrain(rows, cols)
        p_noise = PerlinNoise(octaves=cls.STEPS_FREQUENCY, seed=seed)

        # Calculamos el ruido de Perlin
        noise = np.zeros((rows, cols))
        for i in range(0, rows, width):
            for j in range(0, cols, width):
                noise[i][j] = p_noise([i/rows, j/cols]) 
                noise[i][j] += uniform(-cls.STEPS_NOISE, cls.STEPS_NOISE)

        # Agregamos los bloque siguiendo el ruido de Perlin
        min_noise = np.min(noise)
        for i in range(0, rows, width):
            for j in range(0, cols, width):
                cls.step(terrain, i, j, width, width, height * (noise[i][j] - min_noise))

        return terrain

    @classmethod
    def stairs(cls, rows: int, cols: int, width: int, height: float) -> np.ndarray:
        """
            Genera un terreno de escaleras

            Parametros:
            -----------
            rows: int
                Numero de filas del terreno.
            cols: int
                Numero de columnas del terreno.
            width: int
                Ancho de los escalones.
            height: float
                Altura de los escalones.

            Return:
            -------
            np.ndarray
                Terreno resultante.
        """
        terrain = cls.terrain(rows, cols)

        # Espacio ocupado por la zona central
        middle_width = cls.ZONE_STAIRS_WIDTH

        # Dividimos el terreno en 5 zonas: 3 planas y 2 de escalera. Calculamos el espacio
        # ocupado por las escaleras
        stair_length = (cols - 3 * cls.ZONE_STAIRS_WIDTH) // 2
        middle_width += (cols - 3 * cls.ZONE_STAIRS_WIDTH) % 2

        # Calculamos cuantos escalones toca por escalera
        n = stair_length // width
        middle_width += 2 * (stair_length % width)
        middle_col = cls.ZONE_STAIRS_WIDTH + n * width

        # Calculamos la altura final de las escaleras
        middle_height = (n - 1) * height

        # Generamos las escaleras
        cls.stair(terrain, 0, cls.ZONE_STAIRS_WIDTH, 'E', rows, width, height, n)
        cls.stair(terrain, 0, middle_col + middle_width, 'E', rows, width, height, n)

        # Generamos la zona central
        cls.step(terrain, 0, middle_col, rows, cols, middle_height)

        return terrain

    @staticmethod
    def save(terrain: np.ndarray, filename: str):
        """
            Almacena el terreno en un archivo de texto.

            Parametros:
            -----------
            terrain: numpy.ndarray, shape (M, N)
                Terreno a almacenar.
            filename: str
                Nombre del archivo donde se almacenara el terreno.
        """
        rows, cols = terrain.shape

        # Obtenemos el string que representa al terreno.
        terrain_str = ''
        for i in range(rows):
            for j in range(cols):
                terrain_str += f'{terrain[i][j]}, '
            terrain_str += '\n'

        with open(filename, 'w') as f:
            f.write(terrain_str)

    @staticmethod
    def plot(terrain: np.ndarray):
        """ Genera un grafico del terreno. """
        plt.imshow(terrain, cmap='gray')
        plt.show()


