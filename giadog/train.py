#!/usr/bin/env python3
import json
import pathlib
import argparse
from time import time, sleep
from src.giadog_gym import *
from src.terrain_curriculum import *
try: import rospy
except: pass

from src.agents import *
from src.simulation import *


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


class Rate(object):
    """
        Convenience class for sleeping in a loop at a specified rate

        References:
        -----------
            * https://github.com/strawlab/ros_comm/blob/master/clients/rospy/src/rospy/timer.py
    """
    
    def __init__(self, hz: int):
        """
            Parameters:
            -----------
                * hz: int
                    Rate to determine sleeping
        """
        self.last_time = time()
        self.sleep_dur = 1 / hz

    def sleep(self):
        """
            Attempt sleep at the specified rate. sleep() takes into account the time 
            elapsed since the last successful sleep().
        """
        curr_time = time()

        # Detect time jumping backwards
        if self.last_time > curr_time:
            self.last_time = curr_time

        # Calculate sleep interval
        elapsed = curr_time - self.last_time
        sleep(max(0, self.sleep_dur - elapsed))
        self.last_time = self.last_time + self.sleep_dur

        # Detect time jumping forwards, as well as loops that are inherently too slow
        if curr_time - self.last_time > self.sleep_dur * 2:
            self.last_time = curr_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run \033[1mGiaDog\033[0m train environment.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Simulation arguments
    parser.add_argument(
        '-r', '--ros',
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
        help='The simulation GUI will be displayed.',
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

    args = parser.parse_args()
    
    terrain = steps(ROWS, COLS, 0.7, 0.0, 1)
    save_terrain(terrain, TERRAIN_FILE)

    init_terrain_args = {
        'type'   : 'steps', 
        'rows'   : ROWS, 
        'cols'   : COLS, 
        'width'  : 0.7, 
        'height' : 0.0, 
        'seed'   : 1
    }

    if args.ros:
        rospy.init_node('train', anonymous=True)
        sim = None 
        train_envs = [teacher_giadog_env(sim)]
        train_envs[0].make_terrain(**init_terrain_args)

    else:
        # Initialize simulation
        print('\033[1;36m[i]\033[0m Starting simulations')

        train_envs = []
        for _ in range(args.threads):
            sim = Simulation(args.spot_urdf, gui=args.gui)
            sim.p.setTimeStep(SIM_SECONDS_PER_STEP)
            sim.reset(TERRAIN_FILE, X_INIT, Y_INIT)

            train_envs.append(teacher_giadog_env(sim))
            train_envs[-1].make_terrain(**init_terrain_args)

    print('Running!')
    tc = terrain_curriculum(train_envs)
    tc.train()

