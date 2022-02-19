#!/usr/bin/env python3
import json
import pathlib
import argparse
import pybullet as p
from time import time, sleep
from src.giadog_gym import *
try: import rospy
except: pass

from src.agents import teacher_agent, ars_agent
from src.neural_networks import teacher_nn


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
    
    agent_params = {
        
        "step size": 0.02, # a.k.a. learning rate
        "directions sampled by iteration": 16,
        "exploration standard deviation noise": 0.03, # Debe ser menor a 1
        "number of top directions to use": 16,
        "enviroment": train_env, 
        "train episode steps": 2000, # Numero de steps que cada exploracion tendra H. (N es el nuemro de exploraciones) 
    }

    agent = ars_agent(
            agent_params
    ) #teacher_agent(teacher_nn(), train_env)
    
    print('Running!')
        
    while True:
        # Reseteamos el terreno
        train_env.reset(TERRAIN_FILE)
        # Esperamos que el timestep se reinice
        while train_env.timestep > MAX_ITERATION_TIME: pass

        done = False
        obs = train_env.get_obs()
        rate = Rate(STEPS_PER_REAL_SECOND)

        for i in range(200):
            agent.update_V2(TERRAIN_FILE)
        
        total_r = 0
        while not done:
            # Obtenemos la accion de la politica
            agent.normalizer.observe(agent.process_obs(obs))  
            action = agent.action(obs)
            # Aplicamos la accion al entorno
            print(action)
            obs, reward, done, info = train_env.step([action])
            rate.sleep()
            total_r = total_r + reward
        
        print("total_r", total_r)

        tr = train_env.traverability()
        print(f'Traverability: {tr}')


