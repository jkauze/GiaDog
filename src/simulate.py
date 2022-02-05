#!/usr/bin/env python3
"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains the ROS node in charge of publishing and modifying the status of 
    the spot-mini through ROS topics.
"""
# Utilities
import pathlib
import argparse
from typing import *
from time import time, sleep
from random import randint
from threading import Thread

# Simulation
import rospy
import pybullet as p
from src.simulation import simulation
from src.terrain_gen import terrain_gen
from spot_mini_ros.msg import joint_angles, normal_data, priviliged_data, text

# Rate messages for second
RATE   = 10
# Terrain dimensions
ROWS   = 500
COLS   = 500
# Inicial position
X_INIT = 0.0
Y_INIT = 0.0

def run_simulation(sim: simulation, steps_per_second: int, seconds_per_step: float):
    """ Run the simulation. """
    print('\033[1;36m[i]\033[0m Simulation is running!')
    rate = rospy.Rate(steps_per_second) 
    p.setTimeStep(seconds_per_step)

    # Use the following variables to monitor the number of updates per second.
    begin = time()
    count = 0

    while True: 
        sim.p.stepSimulation()
        rate.sleep()
        count += 1

        # Report simulation speed every thousand steps
        if count % 1000 == 0: 
            vel = count / (time() - begin)
            print(f'Execunting approximately {vel} steps per second.')

def update_sensors(
        steps_per_second: int, 
        data_name: str, 
        update_function: Callable
    ):
    """ Keeps a robot sensor up to date. """
    print(f'\033[1;36m[i]\033[0m Updating {data_name}.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        try: update_function
        except p.error as e: 
            print(f'\033[1;93m[w]\033[0m Warning: {e}')
            sleep(1)
        rate.sleep()

def normal_data_publisher(sim: simulation):
    """
        Function in charge of publishing through a ROS topic the normal data from
        simulation.
    """
    try:
        # Init node
        pub = rospy.Publisher('normal_data', normal_data, queue_size=RATE)

        # Run until manually stopped
        print('\033[1;36m[i]\033[0m Topic "normal_data" is running!')
        rate = rospy.Rate(RATE) 
        while not rospy.is_shutdown():
            # Create message
            msg = normal_data()

            # Velocities
            msg.linear_vel        = list(sim.base_linear_velocity)
            msg.angular_vel       = list(sim.base_angular_velocity)

            # Contact states
            msg.toes_contact      = list(sim.toes_contact_states)
            msg.thighs_contact    = list(sim.thighs_contact_states)
            msg.shanks_contact    = list(sim.shanks_contact_states)

            # Joints states
            msg.joint_angles      = list(sim.joint_angles)
            msg.joint_velocities  = list(sim.joint_velocities)

            # Tranformation matrices
            msg.transform_matrix0 = list(sim.transformation_matrices[0])
            msg.transform_matrix1 = list(sim.transformation_matrices[1])
            msg.transform_matrix2 = list(sim.transformation_matrices[2])
            msg.transform_matrix3 = list(sim.transformation_matrices[3])

            # Publish
            pub.publish(msg)
            rate.sleep()

    except rospy.ROSInterruptException:
        print('\033[1;93m[w]\033[0m Topic "normal_data" was stopped.')

def priviliged_data_publisher(sim: simulation):
    """
        Function in charge of publishing through a ROS topic the priviliged data from 
        simulation.
    """
    try:
        # Init node
        pub = rospy.Publisher('priviliged_data', priviliged_data, queue_size=RATE)

        # Run until manually stopped
        print('\033[1;36m[i]\033[0m Topic "priviliged_data" is running!')
        rate = rospy.Rate(RATE) 
        while not rospy.is_shutdown():
            # Create message
            msg = priviliged_data()

            msg.joint_torques    = list(sim.joint_torques)

            # Normal at each toe
            msg.normal_toe_fl    = list(sim.terrain_normal_at_each_toe[0])
            msg.normal_toe_fr    = list(sim.terrain_normal_at_each_toe[1])
            msg.normal_toe_bl    = list(sim.terrain_normal_at_each_toe[2])
            msg.normal_toe_br    = list(sim.terrain_normal_at_each_toe[3])

            # Force at each toe
            msg.toes_force1      = list(sim.contact_force_at_each_toe)
            msg.toes_force2      = list(sim.toe_force_sensor)

            # Ground friction coefficients at each toe 
            msg.ground_friction  = list(sim.foot_ground_friction_coefficients)

            # Height scan at each toe 
            msg.height_scan_fl   = list(sim.height_scan_at_each_toe[0])
            msg.height_scan_fr   = list(sim.height_scan_at_each_toe[1])
            msg.height_scan_bl   = list(sim.height_scan_at_each_toe[2])
            msg.height_scan_br   = list(sim.height_scan_at_each_toe[3])

            # Publish
            pub.publish(msg)
            rate.sleep()

    except rospy.ROSInterruptException:
        print('\033[1;93m[w]\033[0m Topic "priviliged_data" was stopped.')

def joint_angles_setter(sim: simulation):
    try:
        def actuate_joints(data: joint_angles, sim: simulation=sim):
            """
                Moves the robot's joints to a target position determined by a message 
                received through a ROS topic
            """
            sim.actuate_joints(data.joint_target_positions)

        rospy.Subscriber("spot_joints", joint_angles, actuate_joints)
        print('\033[1;36m[i]\033[0m Topic "spot_joints" is running!')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        print('\033[1;93m[w]\033[0m Topic "spot_joints" was stopped.')

def reset_simulation_subscriber(sim: simulation):
    try:
        def reset(data: text, sim: simulation=sim):
            """
                Reset simulation.
            """
            if data.text == '': terrain_file = sim.terrain_file 
            else: terrain_file = data.text 
            sim.reset(terrain_file, X_INIT, Y_INIT) 

        rospy.Subscriber("reset_simulation", text, reset)
        print('\033[1;36m[i]\033[0m Topic "reset_simulation" is running!')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        print('\033[1;93m[w]\033[0m Topic "reset_simulation" was stopped.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run \033[1mGiaDog\033[0m simulation.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-t', '--terrain_type',
        choices=['hills', 'steps', 'stairs'],
        default='hills',
        help='Type of terrain to generate.',
        metavar='TYPE'
    )

    # Hill arguments
    parser.add_argument(
        '-r', '--roughness',
        type=float,
        default='0.0',
        help='Roughness of hill terrain. It should preferably be in ' +\
            'the range [0, 0.05].',
        metavar='ROUGH'
    )
    parser.add_argument(
        '-f', '--frequency',
        type=float,
        default='0.2',
        help='How often the hills appear. It must be positive, ' +\
            'preferably in the range [0.2, 1].',
        metavar='FREQ'
    )
    parser.add_argument(
        '-a', '--amplitude',
        type=float,
        default='0.2',
        help='Maximum height of the hills.',
        metavar='AMP'
    )

    # Steps and stairs arguments
    parser.add_argument(
        '-w', '--width',
        type=int,
        default='35',
        help='Width of cubes in steps or of the stairs.',
        metavar='WIDTH'
    )
    parser.add_argument(
        '-e', '--height',
        type=float,
        default='0.03',
        help='Max height of cubes in steps or of the ' +\
            'stairs.',
        metavar='HEIGHT'
    )

    # Other terrain arguments
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=randint(0, 1e9),
        help='Specific seed you want to initialize the random generator with.',
        metavar='SEED'
    )
    parser.add_argument(
        '-p', '--predefined_terrain',
        type=str,
        default=None,
        help='Allows to select an already generated terrain stored in a text file.',
        metavar='PATH'
    )

    # Simulation arguments
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
        '--steps-per-second',
        type=int,
        default='100',
        help='Simulation steps per real second.',
        metavar='STEPS'
    )
    parser.add_argument(
        '--seconds-per-step',
        type=float,
        default='0.01',
        help='Simulation seconds for step.',
        metavar='SECONDS'
    )

    args = parser.parse_args()

    # Generating terrain if necessary
    if args.predefined_terrain == None:
        terrain_file = 'terrain_file_tmp.txt'

        if args.terrain_type == 'hills':
            print(
                f'\033[1;36m[i]\033[0m Generating hills with roughness ' +\
                f'{args.roughness}, frequency {args.frequency}, amplitude ' +\
                f'{args.amplitude} and seed {args.seed}.'
            )
            terrain = terrain_gen.hills(
                ROWS, 
                COLS, 
                float(args.roughness), 
                float(args.frequency),
                float(args.amplitude),
                args.seed
            )
            terrain_gen.save(terrain, terrain_file)

        elif args.terrain_type == 'steps':
            print(
                f'\033[1;36m[i]\033[0m Generating steps with width {args.width}, ' +\
                f'height {args.height} and seed {args.seed}.'
            )
            terrain = terrain_gen.steps(
                ROWS, 
                COLS, 
                int(args.width), 
                float(args.height), 
                args.seed
            )
            terrain_gen.save(terrain, terrain_file)

        else:
            print(
                f'\033[1;36m[i]\033[0m Generating stairs with width {args.width} ' +\
                f'and height {args.height}.'
            )
            terrain = terrain_gen.stairs(ROWS, COLS, int(args.width), float(args.height))
            terrain_gen.save(terrain, terrain_file)

    else:
        terrain_file = args.predefined_terrain

    # Initialize simulation
    print('\033[1;36m[i]\033[0m Starting simulation')
    sim = simulation(args.spot_urdf, bullet_server=p, gui=args.gui)
    sim.reset(terrain_file, X_INIT, Y_INIT) 

    # Run rosnode for spot-mini
    rospy.init_node('spot_simulation', anonymous=True)

    # Thread that runs the simulation
    th_args = (sim, args.steps_per_second, args.seconds_per_step)
    sim_th = Thread(target=run_simulation, args=th_args)
    sim_th.daemon = True
    sim_th.start()

    # Threads that constantly update each sensor
    sensores_ths = []
    update_functions = {
        (sim.update_base_velocity, 'velocities'),
        (sim.update_toes_contact_info, 'toes contact'),
        (sim.update_thighs_contact_info, 'thighs contact'),
        (sim.update_shanks_contact_info, 'shanks contact'),
        (sim.update_height_scan, 'heigh scan'),
        (sim.update_toes_force, 'toes force'),
        (sim.update_joints_sensors, 'joints'),
        (sim.update_transformation_matrices, 'transformation matrices')
    }

    for f, data in update_functions:
        sensores_ths.append(Thread(
            target=update_sensors, 
            args=(args.steps_per_second, data, f)
        ))
    for th in sensores_ths: 
        th.daemon = True
        th.start()

    # Thread that publish the value of each sensor in ROS.
    pub_th = Thread(target=normal_data_publisher, args=(sim,))
    pub_th.daemon = True
    pub_th.start()
    pub_th = Thread(target=priviliged_data_publisher, args=(sim,))
    pub_th.daemon = True
    pub_th.start()

    # Thread that receives data through ROS to update the joints.
    joint_th = Thread(target=joint_angles_setter, args=(sim,))
    joint_th.daemon = True
    joint_th.start()

    # Thread in charge of receiving the signals to restart the simulation. 
    reset_th = Thread(target=reset_simulation_subscriber, args=(sim,))
    reset_th.daemon = True
    reset_th.start()

    # The node keeps runing until the process is canceled
    rospy.spin()
