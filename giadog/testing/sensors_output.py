import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from src.__env__ import TERRAIN_FILE
from src.simulation.Simulation import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test terrain curriculum.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-u', '--spot-urdf',
        type=str,
        default=str(pathlib.Path(__file__).parent.parent.resolve()) +\
            '/mini_ros/urdf/spot.urdf',
        help='Path to the URDF file of the quadruped robot.',
        metavar='PATH'
    )
    parser.add_argument(
        '-s', '---sensor',
        choices=[
            'position-orientation',
            'base-velocity',
            'joints-data',
            'toes-contact'
        ],
        default='position-orientation',
        help='Sensor to test.',
        metavar='SENSOR'
    )

    args = parser.parse_args()

    sim = Simulation(args.spot_urdf, gui=True)
    sim.reset(TERRAIN_FILE)
    
    if args.sensor == 'position-orientation': 
        test_function = sim.test_position_orientation
    elif args.sensor == 'base-velocity':
        test_function = sim.test_base_velocity
    elif args.sensor == 'joints-data':
        test_function = sim.test_joint_sensors
    elif args.sensor == 'toes-contact':
        test_function = sim.test_toes_contact

    sim.test(test_function)