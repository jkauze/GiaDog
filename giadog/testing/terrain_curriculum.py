import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from src.simulation import Simulation
from src.training import GiadogEnv, TerrainCurriculum

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
        '-g', '--gui',
        action='store_const',
        default=False, 
        const=True, 
        help='The simulation GUI will be displayed. In this case only one ' +\
            'thread will be used ignoring the -t|--threads flag.',
    )
    parser.add_argument(
        '-t', '---type',
        choices=['hills', 'steps', 'stairs'],
        default='hills',
        help='Terrain type.',
        metavar='TYPE'
    )
    parser.add_argument(
        '-r', '--roughness',
        type=float,
        default='0.0',
        help='Roughness of the terrain. Preferably in the range [0.0, 0.05].',
        metavar='ROUGHNESS'
    )
    parser.add_argument(
        '-f', '--frequency',
        type=float,
        default='0.2',
        help='How often the hills appear. It must be positive, preferably ' +\
            'in the range [0.2, 2.0].',
        metavar='FREQUENCY'
    )
    parser.add_argument(
        '-a', '--amplitude',
        type=float,
        default='0.2',
        help='Maximum height of the hills. It should preferably be in the ' +\
            'range [0.2, 2.0].',
        metavar='AMPLITUDE'
    )
    parser.add_argument(
        '-w', '--width',
        type=float,
        default='0.8',
        help='Width and length of the steps. It should preferably be in the ' +\
            'range [0.3, 0.8].',
        metavar='WIDTH'
    )
    parser.add_argument(
        '-H', '--height',
        type=float,
        default='0.02',
        help='Maximum height of the steps. It should preferably be in the ' +\
            'range [0.05, 0.4] for steps or [0.02, 0.1] for stairs.',
        metavar='HEIGHT'
    )
    parser.add_argument(
        '-s', '--show-every',
        type=int,
        default='50',
        help='Indicates every how many epochs the terrain corresponding to ' +\
            'the best particle should be shown. This flag should be used ' +\
            'together with the -g|--gui flag',
        metavar='EPOCHS'
    )

    args = parser.parse_args()

    sim = Simulation(args.spot_urdf, gui=args.gui)

    tc = TerrainCurriculum(
        [GiadogEnv(sim)], 
        train_method='Test',
        _continue=False, 
        type=args.type,
        epoch_to_show=args.epoch_to_show,
        roughness=args.roughness,
        frequency=args.frequency,
        amplitude=args.amplitude,
        width=args.width,
        height=args.height
    )

    tc.train()
