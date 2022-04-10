#!/usr/bin/env python3
import json
import pathlib
import argparse
from src.simulation import Simulation
from src.training import GiadogEnv, TerrainCurriculum
try: import rospy
except: pass



# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)
# Obtenemos las constantes necesarias
ROWS                  = ENV["SIMULATION"]["ROWS"]
COLS                  = ENV["SIMULATION"]["COLS"]
X_INIT                = ENV["SIMULATION"]["X_INIT"]
Y_INIT                = ENV["SIMULATION"]["Y_INIT"]
TERRAIN_FILE          = ENV["SIMULATION"]["TERRAIN_FILE"]
STEPS_PER_REAL_SECOND = ENV["SIMULATION"]["STEPS_PER_REAL_SECOND"]
SIM_SECONDS_PER_STEP  = ENV["SIMULATION"]["SIM_SECONDS_PER_STEP"]
MAX_ITERATION_TIME    = ENV["TRAIN"]["MAX_ITERATION_TIME"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run \033[1mGiaDog\033[0m train environment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation arguments
    parser.add_argument(
        '--ros',
        action='store_const',
        default=False, 
        const=True, 
        help='The simulation is running using ROS.',
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
        '-m', '---method',
        choices=['TRPO', 'PPO', 'ARS'],
        default='TRPO',
        help='Training method.',
        metavar='METHOD'
    )
    parser.add_argument(
        '-u', '--spot-urdf',
        type=str,
        default=str(pathlib.Path(__file__).parent.resolve()) +\
            '/mini_ros/urdf/spot.urdf',
        help='Path to the URDF file of the quadruped robot.',
        metavar='PATH'
    )
    parser.add_argument(
        '-t', '--threads',
        type=int,
        default='1',
        help='Number of threads to parallelize execution. It must be positive. It ' +\
            'can only be greater than 1 if the -r,--ros flag is not used.',
        metavar='THREADS'
    )
    parser.add_argument(
        '-r', '--resume',
        action='store_const',
        default=False, 
        const=True, 
        help='Indicates that the training should be continued instead of ' +\
            'starting a new one.',
    )

    args = parser.parse_args()
    
    print(f'\033[1;36m[i]\033[0m # {"="*20} STARTING TRAINING {"="*20} #')

    if args.gui: args.threads = 1

    if args.ros:
        rospy.init_node('train', anonymous=True)
        sim = None 
        train_envs = [GiadogEnv(sim)]

    else:
        # Initialize simulation
        print(f'\033[1;36m[i]\033[0m Starting {args.threads} simulations...')

        train_envs = []
        for _ in range(args.threads):
            sim = Simulation(args.spot_urdf, gui=args.gui)
            sim.p.setTimeStep(SIM_SECONDS_PER_STEP)
            train_envs.append(GiadogEnv(sim))

    tc = TerrainCurriculum(train_envs, args.method, args.resume)
    tc.train()

