"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    File containing the code in charge of the automatic generation of simulated 
    terrain.
"""
import numpy as np
from typing import *
from random import uniform
import matplotlib.pyplot as plt
from perlin_noise import PerlinNoise
from src.__env__ import SCALE, STEPS_FREQUENCY, STEPS_NOISE, ZONE_STAIRS_WIDTH


def __terrain(rows: int, cols: int) -> np.ndarray:
    """
        Generate a new flat terrain.

        Parameters:
        -----------
            rows: int
                Number of rows.

            cols: int
                Number of columns.

        Return:
        -------
            numpy.ndarray, shape (rows, cols)
                Flat terrain of dimensions rows x cols.
    """
    return np.zeros((rows, cols))

def __perlin(terrain: np.ndarray, amplitude: float, octaves: float, seed: int):
    """
        Apply the Perlin noise on a terrain.

        Parameters:
        -----------
            terrain: numpy.ndarray, shape (M, N)
                Terrain to modify.

            amplitude: float
                Maximum noise amplitude.

            octaves: float
                Number of sub rectangles in each range [0, 1].
                
            seed: int 
                Specific seed you want to initialize the random generator with.
    """
    p_noise = PerlinNoise(octaves=octaves, seed=seed)
    rows, cols = terrain.shape

    # Calculate the noise.
    noise = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            noise[i][j] = p_noise([i/rows, j/cols])

    # Apply the noise to the terrain.
    terrain += amplitude * (noise - np.min(noise))

def __step(
        terrain: np.ndarray, 
        row: int, 
        col: int, 
        width: int, 
        lenght: int, 
        height: float
    ):
    """
        Add a cube to the terrain.

        Parameters:
        -----------
            terrain: numpy.ndarray, shape (M, N)
                Terrain to modify.

            row: int
                Row in which the upper left corner of the cube is located.

            col: int
                Column in which the upper left corner of the cube is located.

            width: int
                Width (number of rows) that the cube occupies.

            lenght: int
                Length (number of columns) that the cube occupies.
                
            height: float
                Cube height.
    """
    rows, cols = terrain.shape
    for i in range(row, min(rows, row + width)):
        for j in range(col, min(cols, col + lenght)):
            terrain[i][j] += height

def __stair(
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
        Place a stair on the terrain.

        Parameters:
        -----------
            terrain: numpy.ndarray, shape (M, N)
                Terrain to modify.

            row: int
                Top row where the corner of the lowest step is located.

            col: int
                Upper column where the corner of the lowest step is located.

            orientation: str, {'E', 'S', 'W', 'N'}
                Orientation of the stair.

            width: int
                Steps width.

            length: int
                Steps length.

            height: float
                Steps height.
                
            n: int
                Number of steps.
    """
    if orientation == 'E':
        for i in range(n): 
            __step(
                terrain, 
                row, 
                col + i * length, 
                width, 
                length, 
                i * height
            )
    elif orientation == 'S':
        for i in range(n): 
            __step(
                terrain, 
                row + i * width, 
                col,
                 width, 
                 length, 
                 i * height
                )
    elif orientation == 'W':
        for i in range(n):
            __step(
                terrain, 
                row, 
                col - (i + 1) * length, 
                width, 
                length, 
                i * height
            )
    elif orientation == 'N':
        for i in range(n):
            __step(
                terrain, 
                row - (i + 1)* width, 
                col, 
                width, 
                length, 
                i * height
            )
    else:
        raise Exception(f'Unexpected orientation "\033[1;3m{orientation}\033[0m"')

def __goal(terrain: np.ndarray, row: int, col: int, height: float):
    """
        Place a vertical rod on the terrain.

        Parameters:
        -----------
            terrain: np.ndarray, shape (M, N)  
                Terrain to modify.

            row: int
                Row where the rod is located.

            col: int
                Column where the rod is located.
                
            heihgt: int
                Rod height.
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

def set_goal(terrain: np.ndarray, height: float) -> Tuple[int, int]:
    """
        Place a vertical rod randomly at some point on the circumference 
        inscribed on the terrain.
    """
    rows, cols = terrain.shape
    radio = min(rows, cols) // 2 - 2

    # Choose a random angle
    angle = uniform(0, 2 * np.pi)
    row = int(radio * (1 + np.cos(angle)))
    col = int(radio * (1 + np.sin(angle)))

    # Set the goal
    __goal(terrain, row, col, height)

    return (row, col)

def hills(
        rows: int,
        cols: int,
        roughness: float,
        frequency: float, 
        amplitude: float, 
        seed: int
    ) -> np.ndarray:
    """
        Generates a rugged hilly terrain.

        Parameters:
        -----------
            rows: int
                Number of rows of the terrain.

            cols: int
                Number of columns of the terrain.

            roughness: float
                Roughness of the terrain. It should preferably be in the range 
                [0, 0.05].

            frequency: float
                How often the hills appear. It must be positive, preferably in 
                the range [0.2, 1].

            amplitude: float
                Maximum height of the hills.
                
            seed: int 
                Specific seed you want to initialize the random generator with.

        Return:
        -------
            np.ndarray
                Resulting terrain.
    """
    # Generate the terrain
    terrain = __terrain(rows, cols)
    __perlin(terrain, amplitude, frequency, seed)

    # Add the roughness
    for i in range(rows):
        for j in range(cols):
            terrain[i][j] += uniform(-roughness, roughness)

    return terrain

def steps(
        rows: int, 
        cols: int, 
        width: float, 
        height: float, 
        seed: int
    ) -> np.ndarray:
    """
        Generate a cubes terrain. 

        Parameters:
        -----------
            rows: int
                Number of rows of the terrain.

            cols: int
                Number of columns of the terrain.

            width: float
                Width and length of the cubes.

            height: float
                Maximum height of the cubes.
                
            seed: int 
                Specific seed you want to initialize the random generator with.

        Return:
        -------
            np.ndarray
                Resulting terrain.
    """
    width = int(width / SCALE)

    # Generate the terrain
    terrain = __terrain(rows, cols)
    p_noise = PerlinNoise(octaves=STEPS_FREQUENCY, seed=seed)

    # Calculate the Perlin noise
    noise = np.zeros((rows, cols))
    for i in range(0, rows, width):
        for j in range(0, cols, width):
            noise[i][j] = p_noise([i/rows, j/cols]) 
            noise[i][j] += uniform(-STEPS_NOISE, STEPS_NOISE)

    # Add the blocks following the Perlin noise
    min_noise = np.min(noise)
    for i in range(0, rows, width):
        for j in range(0, cols, width):
            __step(
                terrain, 
                i, 
                j, 
                width, 
                width, 
                height * (noise[i][j] - min_noise)
            )

    return terrain

def stairs( 
        rows: int, 
        cols: int, 
        width: float, 
        height: float,
        seed: int=0
    ) -> np.ndarray:
    """
        Generate a terrain of stairs.

        Parameters:
        -----------
            rows: int
                Number of rows of the terrain.

            cols: int
                Number of columns of the terrain.

            width: float
                Steps width.

            height: float
                Steps height.

            seed: int
                Dummy argument.

        Return:
        -------
            np.ndarray
                Resulting terrain.
    """
    terrain = __terrain(rows, cols)
    width = int(width / SCALE)

    # Space occupied by the central area
    middle_width = ZONE_STAIRS_WIDTH

    # We divide the terrain into 5 zones: 3 flat and 2 for stairs. Calculate the
    # space occupied by the stairs
    stair_length = (cols - 3 * ZONE_STAIRS_WIDTH) // 2
    middle_width += (cols - 3 * ZONE_STAIRS_WIDTH) % 2

    # Calculate how many steps each stair has
    n = stair_length // width
    middle_width += 2 * (stair_length % width)
    middle_col = ZONE_STAIRS_WIDTH + n * width

    # Calculate the height of the central zone
    middle_height = (n - 1) * height

    # Generate the stairs
    __stair(terrain, 0, ZONE_STAIRS_WIDTH, 'E', rows, width, height, n)
    __stair(terrain, 0, middle_col + middle_width, 'E', rows, width, height, n)

    # Generate the central zone
    __step(terrain, 0, middle_col, rows, cols, middle_height)
    # Generate the final zone
    __step(terrain, 0, cols - ZONE_STAIRS_WIDTH, rows, cols, (n - 1) * height)

    return terrain

def save_terrain(terrain: np.ndarray, filename: str):
    """
        Stores the terrain in a text file.

        Parameters:
        -----------
        terrain: numpy.ndarray, shape (M, N)
            Terrain to store.
        filename: str
            Name of the file where the terrain will be stored.
    """
    rows, cols = terrain.shape

    # Obtain the string that represents the terrain.
    terrain_str = ''
    for i in range(rows):
        for j in range(cols):
            terrain_str += f'{terrain[i][j]}, '
        terrain_str += '\n'

    with open(filename, 'w') as f:
        f.write(terrain_str)

def plot(terrain: np.ndarray):
    """ Generate a plot of the terrain. """
    plt.imshow(terrain, cmap='gray')
    plt.show()


