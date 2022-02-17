#!/usr/bin/env python3
import json
import rospy
import pathlib
import argparse
import pybullet as p
from src.giadog_gym import *


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

    args = parser.parse_args()
    
    terrain = terrain_gen.steps(ROWS, COLS, 0.7, 0.0, 1)
    terrain_gen.save(terrain, TERRAIN_FILE)

    if args.ros:
        rospy.init_node('train', anonymous=True)
        sim = None 
    else:
        # Initialize simulation
        print('\033[1;36m[i]\033[0m Starting simulation')
        sim = simulation(args.spot_urdf, bullet_server=p, gui=args.gui)
        sim.p.setTimeStep(SIM_SECONDS_PER_STEP)
        sim.reset(TERRAIN_FILE, X_INIT, Y_INIT)

    train_env = teacher_giadog_env(sim)
    train_env.make_terrain('steps', rows=ROWS, cols=COLS, width=0.7, height=0.0, seed=1)

    print('Running!')
    while True:
        # Reseteamos el terreno
        train_env.reset(TERRAIN_FILE)
        # Esperamos que el timestep se reinice
        while train_env.timestep > MAX_ITERATION_TIME: pass

        done = False
        obs = train_env.get_obs()
        while not done:
            # Obtenemos la accion de la politica
            action = train_env.predict(obs)
            # Aplicamos la accion al entorno
            obs, reward, done, info = train_env.step(action)

        tr = train_env.traverability()
        print(f'Traverability: {tr}')


