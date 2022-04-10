import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

from Simulation import Simulation
from terrain_gen import hills, steps, stairs, set_goal, save_terrain, plot_terrain
