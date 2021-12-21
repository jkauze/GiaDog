"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""

# Utilidades
from typing import *
from sys import argv, stderr
import time

# Simulacion
import pybullet as pb
import pybullet_data as pd

from terrain_gen import terrain_gen

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

            Parametros:
            -----------
            terrain_file: str
                Archivo que contiene el terreno.
            giadog_file: str
                Archivo que contiene la especificacion del agente.
            mesh_scale: List[int], optional
                Lista 3D que representa la scala del terreno. 
                Default: [1/50, 1/50, 1]
            init: Tuple[int, int], optional
                Coordenada inicial. La altura es calculada de forma automatica. 
                Default: (0, 0)
        """
        # Obtenemos el numero de filas y columnas
        h = 5

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
    terrain = terrain_gen.hills(500, 500, 0.04, 3, 3, 42)
    terrain_gen.set_goal(terrain, 2)
    terrain_gen.save(terrain, argv[1])
    simulation().test_terrain(argv[1], 'src/mini_ros/urdf/spot.urdf')