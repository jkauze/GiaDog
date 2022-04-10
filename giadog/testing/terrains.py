import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import argparse
from random import randint
from src.simulation import hills, steps, stairs, plot_terrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test terrain generation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        '-s', '--seed',
        type=int,
        default=None,
        help='Specific seed you want to initialize the random generator with.',
        metavar='SEED'
    )

    args = parser.parse_args()

    if args.seed == None: args.seed = randint(0, 1e10)

    rows = 500
    cols = 500
    if args.type == 'hills': 
        terrain = hills(
            rows, 
            cols, 
            args.roughness, 
            args.frequency, 
            args.amplitude,
            args.seed 
        )
    elif args.type == 'steps':
        terrain = steps(
            rows, 
            cols, 
            args.width,
            args.height,
            args.seed 
        )
    else:
        terrain = stairs(
            rows, 
            cols, 
            args.width,
            args.height,
            args.seed 
        )

    plot_terrain(terrain)