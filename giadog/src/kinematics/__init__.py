import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

from inverse_kinematics import solve_leg_IK
from FTG import FTG, FTG_debug, foot_trajectories, foot_trajectories_debug
from transformation_matrices import transformation_matrices, \
    rotation_matrix_from_euler