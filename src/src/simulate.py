#!/usr/bin/env python3
"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains the ROS node in charge of publishing and modifying the status of 
    the spot-mini through ROS topics.
"""
# Utilities
import pathlib
from sys import argv
from time import time
from random import randint
from threading import Thread
from getopt import getopt, GetoptError 

# Simulation
import rospy
import pybullet as p
from simulation import simulation
from terrain_gen import terrain_gen
from spot_mini_ros.msg import spot_state, joint_angles

# Rate messages for second
RATE   = 10
# Terrain dimensions
ROWS   = 500
COLS   = 500
# Inicial position
X_INIT = 0.0
Y_INIT = 0.0
# Steps per real second.
STEPS_PER_SECOND = 120
# Simulate seconds for step.
SECONDS_PER_STEP = 1/60

def run_simulation(sim: simulation, steps_per_second: int, seconds_per_step: float):
    """ Run the simulation. """
    print('[i] Simulation is running!')
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

def update_base_velocity(sim: simulation, steps_per_second: int):
    """ Updates the base linear and angular velocity for the current simulation step. """
    print('[i] Updating velocities.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_base_velocity()
        rate.sleep()

def update_toes_contact_info(sim: simulation, steps_per_second: int):
    """
        Updates the contact info for each toe for the current simulation steps. The
        contact info include:
            * terrain_normal_at_each_toe
            * contact_force_at_each_toe
            * foot_ground_friction_coefficients
            * toes_contact_states
    """
    print('[i] Updating toes contact info.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_toes_contact_info()
        rate.sleep()

def update_thighs_contact_info(sim: simulation, steps_per_second: int):
    """ Updates the contact info for each thigh for the current simulation step. """
    print('[i] Updating thighs contact info.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_thighs_contact_info()
        rate.sleep()

def update_shanks_contact_info(sim: simulation, steps_per_second: int):
    """
        Updates the contact info for each shank for the current simulation step.
    """
    print('[i] Updating shanks contact info.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_shanks_contact_info()
        rate.sleep()

def update_height_scan(sim: simulation, steps_per_second: int):
    """
        Update the height scan for each step for the current simulation step.
    """
    print('[i] Updating heigh scan.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_height_scan()
        rate.sleep()

def update_toes_force(sim: simulation, steps_per_second: int):
    """
        Update force in each step for the current simulation step.
    """
    print('[i] Updating toes force.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_toes_force()
        rate.sleep()

def update_joints_sensors(sim: simulation, steps_per_second: int):
    """
        Update position, velocity and torque for each joint for the current
        simulation step.
    """
    print('[i] Updating joint sensors.')
    rate = rospy.Rate(steps_per_second) 
    while True: 
        sim.update_joints_sensors()
        rate.sleep()

def spot_state_publisher(sim: simulation):
    """
        Function in charge of publishing through a ROS topic the data from simulation.
    """
    try:
        # Init node
        pub = rospy.Publisher('spot_state', spot_state, queue_size=RATE)

        # Run until manually stopped
        print('[i] Topic "spot_state" is running!')
        rate = rospy.Rate(RATE) 
        while not rospy.is_shutdown():
            # Create message
            msg = spot_state()

            # Velocities
            msg.linear_vel       = list(sim.base_linear_velocity)
            msg.angular_vel      = list(sim.base_angular_velocity)

            # Contact states
            msg.toes_contact     = list(sim.toes_contact_states)
            msg.thighs_contact   = list(sim.thighs_contact_states)
            msg.shanks_contact   = list(sim.shanks_contact_states)

            # Joints states
            msg.joint_angles     = list(sim.joint_angles)
            msg.joint_velocities = list(sim.joint_velocities)

            # ===================== Privileged Data ===================== #
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
        print('[w] Topic "spot_state" was stopped.')

def joint_angles_setter(sim: simulation):
    try:
        def actuate_joints(data, sim: simulation=sim):
            """
                Moves the robot's joints to a target position determined by a message 
                received through a ROS topic
            """
            sim.actuate_joints(data.joint_target_positions)

        rospy.Subscriber("spot_joints", joint_angles, actuate_joints)
        print('[i] Topic "spot_joints" is running!')

        # spin() simply keeps python from exiting until this node is stopped
        rospy.spin()
    except rospy.ROSInterruptException:
        print('[w] Topic "spot_joints" was stopped.')

def usage():
    script = '\033[1mrosrun spot_mini_ros simulate.py\033[0m'
    opt = '[-g] [-u|--spot-urdf \033[4mURDF_FILE\033[0m] ' +\
        '[--steps-per-second \033[4mSTEPS\033[0m] ' +\
        '[--seconds-per-step \033[4mSECONDS\033[0m] ' +\
        '[-s|--seed \033[3mSEED\033[0m]'
    print(
        'Usage:\n' +\
        f'    {script} [OPTIONS] \033[4mTERRAIN_TYPE\033[0m \033[4mTERRAIN_ARGS\033[0m\n' +\
        f'    {script} [OPTIONS] -t|--terrain \033[4mTERRAIN_FILE\033[0m\n' +\
        f'    {script} -h|--help\n\n' +\
        'Options:\n' +\
        '    \033[1m-g\033[0m    The simulation GUI will be displayed.\n'
        '    \033[1m-u\033[0m|\033[1m--spot-urdf\033[0m \033[4mURDF_FILE\033[0m    ' +\
            'Path to the URDF file of the quadruped robot.\n' +\
        '    \033[1m--steps-per-second\033[0m \033[4mSTEPS\033[0m   ' +\
            'Simulation steps per real second. STEPS must bu integer.\n' +\
        '    \033[1m--seconds-per-step\033[0m \033[4mSECONDS\033[0m   ' +\
            'Simulation seconds for step. SECONDS must be floating.\n' +\
        '   \033[1m-s\033[0m|\033[1m--seed\033[0m \033[3mSEED\033[0m    ' +\
            'Specific seed you want to build terrains.'
    )

if __name__ == '__main__':
    # Get arguments
    try:
        opts, args = getopt(
            argv[1:], 
            'hgt:u:s:', 
            ['help', 'terrain=', 'spot-urdf=', 'steps-per-second=', 'seconds-per-step=', 'seed=']
        )
    except GetoptError as err:
        print(err)
        usage()
        exit(2)

    # Default values
    gui = False
    custom_terrain = False 
    seed = randint(0, 1e9)
    steps_per_second = STEPS_PER_SECOND
    seconds_per_step = SECONDS_PER_STEP
    giadog_urdf_file = str(pathlib.Path(__file__).parent.parent.resolve()) 
    giadog_urdf_file += '/mini_ros/urdf/spot.urdf'

    # Processing arguments
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            usage()
            exit()
        elif opt == '-g':
            gui = True
        elif opt in ('-t', '--terrain'):
            custom_terrain = True 
            terrain_file = arg
        elif opt in ('-u', '--spot-urdf'):
            giadog_urdf_file = arg
        elif opt == '--steps-per-second':
            steps_per_second = int(arg)
        elif opt == '--seconds-per-step':
            seconds_per_step = float(arg)
        elif opt in ('-s', '--seed'):
            seed = int(arg)
        else:
            raise Exception(f'Unhandled option \033[1;3m{opt}\033[0m.')

    # Generating terrain if necessary
    if not custom_terrain:
        if len(args) < 3:
            usage()
            exit(1)

        terrain_file = 'terrain_file_tmp.txt'
        
        if args[0] == 'hills':
            if len(args) < 4:
                #usage()
                exit(1)

            roughness, frequency, amplitude = args[1:4]

            print(
                f'[i] Generating hills with roughness {roughness}, ' +\
                f'frequency {frequency}, amplitude {amplitude} and seed {seed}.'
            )
            terrain = terrain_gen.hills(
                ROWS, 
                COLS, 
                float(roughness), 
                float(frequency),
                float(amplitude),
                seed
            )
            terrain_gen.save(terrain, terrain_file)

        elif args[0] == 'steps':
            width, height = args[1:3]

            print(
                f'[i] Generating steps with width {width}, height {height} ' +\
                f'and seed {seed}.'
            )
            terrain = terrain_gen.steps(ROWS, COLS, int(width), float(height), seed)
            terrain_gen.save(terrain, terrain_file)

        elif args[0] == 'stairs':
            width, height = args[1:3]

            print(f'[i] Generating stairs with width {width} and height {height}.')
            terrain = terrain_gen.stairs(ROWS, COLS, int(width), float(height))
            terrain_gen.save(terrain, terrain_file)

        else:
            raise Exception(f'Undefined terrain type \033[1;3m{args[0]}\033[0m.')

    # Initialize simulation
    print('[i] Starting simulation')
    sim = simulation(terrain_file, giadog_urdf_file, bullet_server=p)
    sim.initialize(X_INIT, Y_INIT, gui) 

    # Run rosnode for spot-mini
    rospy.init_node('spot_state_node', anonymous=True)

    # Thread that runs the simulation
    args = (sim, steps_per_second, seconds_per_step)
    sim_th = Thread(target=run_simulation, args=args)
    sim_th.start()

    # Threads that constantly update each sensor
    args = (sim, steps_per_second)
    sensores_ths = []
    sensores_ths.append(Thread(target=update_base_velocity, args=args))
    sensores_ths.append(Thread(target=update_toes_contact_info, args=args))
    sensores_ths.append(Thread(target=update_thighs_contact_info, args=args))
    sensores_ths.append(Thread(target=update_shanks_contact_info, args=args))
    sensores_ths.append(Thread(target=update_height_scan, args=args))
    sensores_ths.append(Thread(target=update_toes_force, args=args))
    sensores_ths.append(Thread(target=update_joints_sensors, args=args))
    for th in sensores_ths: th.start()

    # Thread that publish the value of each sensor in ROS.
    pub_th = Thread(target=spot_state_publisher, args=(sim,))
    pub_th.start()

    # Thread that receives data through ROS to update the joints.
    joint_th = Thread(target=joint_angles_setter, args=(sim,))
    joint_th.start()
