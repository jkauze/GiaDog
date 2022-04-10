import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

from FTG import FTG, foot_trajectories
from inverse_kinematics import solve_leg_IK
from transformation_matrices import transformations_matrices