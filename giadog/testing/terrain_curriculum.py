import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from src.simulation.Simulation import *
from src.training.TerrainCurriculum import *

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
        [TeacherEnv(sim)], 
        train_method='',
        _continue=False, 
        testing=True
    )
    
    def artificial_trajectory_gen(p: Particle, k: int, m: int):
        if p.type != args.type: return 

        mean = (MAX_DESIRED_TRAV + MIN_DESIRED_TRAV) / 2
        std = 0.0625

        if p.type == 'hills':
            fitness = abs(args.roughness - p.parameters['roughness']) / \
                (HILLS_RANGE['roughness'][1] - HILLS_RANGE['roughness'][0])
            fitness += abs(args.frequency - p.parameters['frequency']) / \
                (HILLS_RANGE['frequency'][1] - HILLS_RANGE['frequency'][0])
            fitness += abs(args.amplitude - p.parameters['amplitude']) / \
                (HILLS_RANGE['amplitude'][1] - HILLS_RANGE['amplitude'][0])
            fitness /= 3
        elif p.type == 'steps':
            fitness = abs(args.width - p.parameters['width']) / \
                (STEPS_RANGE['width'][1] - STEPS_RANGE['width'][0])
            fitness += abs(args.height - p.parameters['height']) / \
                (STEPS_RANGE['height'][1] - STEPS_RANGE['height'][0])
            fitness /= 2
        else:
            fitness = abs(p.parameters['width'] - args.width) / \
                (STAIRS_RANGE['width'][1] - STAIRS_RANGE['width'][0])
            fitness += abs(args.height - p.parameters['height']) / \
                (STAIRS_RANGE['height'][1] - STAIRS_RANGE['height'][0])
            fitness /= 2

        mean += fitness
        p.traverability[k * N_TRAJ + m] = np.clip(np.random.normal(mean, std), 0, 1)

    tc.train(artificial_trajectory_gen, args.type, args.show_every)
