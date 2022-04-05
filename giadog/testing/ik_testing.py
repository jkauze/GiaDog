import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from src.__env__ import TERRAIN_FILE
from src.simulation.Simulation import *



if __name__ == '__main__':
    sim = Simulation(str(pathlib.Path(__file__).parent.parent.resolve())
                        +'/mini_ros/urdf/spot.urdf', gui=True)
    sim.reset('terrains/gym_terrain.txt', 
                fix_robot_base=True)
    
    test_function = sim.test_IK

    sim.test(test_function)