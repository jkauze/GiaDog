import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from uuid import uuid4
from random import randint
from src.simulation import Simulation, hills, steps, stairs, save_terrain

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test terrain simulation.',
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

    args = parser.parse_args()

    sim = Simulation(args.spot_urdf, gui=True)

    ROWS = COLS = 500

    # Add sliders 
    # Hill
    rough_id = sim.p.addUserDebugParameter('Roughness', 0.0, 0.05)
    freq_id = sim.p.addUserDebugParameter('Frequency', 0.2, 2.5)
    amp_id = sim.p.addUserDebugParameter('Amplitude', 0.2, 2.5)
    hills_gen_id = sim.p.addUserDebugParameter('Generate Hills', 1, 0)

    # Steps
    steps_width_id = sim.p.addUserDebugParameter('Width', 0.3, 0.8)
    steps_height_id = sim.p.addUserDebugParameter('Height', 0.05, 0.4)
    steps_gen_id = sim.p.addUserDebugParameter('Generate Steps', 1, 0)

    # Stairs
    stairs_width_id = sim.p.addUserDebugParameter('Width', 0.3, 0.8)
    stairs_height_id = sim.p.addUserDebugParameter('Height', 0.02, 0.1)
    stairs_gen_id = sim.p.addUserDebugParameter('Generate Stairs', 1, 0)

    hills_count = steps_count = stairs_count = 0
    while True: 
        if hills_count < sim.p.readUserDebugParameter(hills_gen_id):
            print(
                sim.p.readUserDebugParameter(rough_id),
                sim.p.readUserDebugParameter(freq_id),
                sim.p.readUserDebugParameter(amp_id),
            )
            terrain = hills(
                ROWS,
                COLS,
                sim.p.readUserDebugParameter(rough_id),
                sim.p.readUserDebugParameter(freq_id),
                sim.p.readUserDebugParameter(amp_id),
                randint(0, 1e6)
            )
            file = f'./terrains/hills_{uuid4()}.txt'
            save_terrain(terrain, file)
            sim.reset(file)

            hills_count = sim.p.readUserDebugParameter(hills_gen_id)
            steps_count = sim.p.readUserDebugParameter(steps_gen_id)
            stairs_count = sim.p.readUserDebugParameter(stairs_gen_id)
            
        elif steps_count < sim.p.readUserDebugParameter(steps_gen_id):
            terrain = steps(
                ROWS,
                COLS,
                sim.p.readUserDebugParameter(steps_width_id),
                sim.p.readUserDebugParameter(steps_height_id),
                randint(0, 1e6)
            )
            file = f'./terrains/steps_{uuid4()}.txt'
            save_terrain(terrain, file)
            sim.reset(file)

            hills_count = sim.p.readUserDebugParameter(hills_gen_id)
            steps_count = sim.p.readUserDebugParameter(steps_gen_id)
            stairs_count = sim.p.readUserDebugParameter(stairs_gen_id)

        elif stairs_count < sim.p.readUserDebugParameter(stairs_gen_id):
            terrain = stairs(
                ROWS,
                COLS,
                sim.p.readUserDebugParameter(stairs_width_id),
                sim.p.readUserDebugParameter(stairs_height_id)
            )
            file = f'./terrains/stairs_{uuid4()}.txt'
            save_terrain(terrain, file)
            sim.reset(file)
            
            hills_count = sim.p.readUserDebugParameter(hills_gen_id)
            steps_count = sim.p.readUserDebugParameter(steps_gen_id)
            stairs_count = sim.p.readUserDebugParameter(stairs_gen_id)
    