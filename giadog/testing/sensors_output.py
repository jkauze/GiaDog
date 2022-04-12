import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
import argparse
from src.simulation import Simulation

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
        '-r', '--real-step',
        action='store_const',
        default=False, 
        const=True, 
        help='',
    )
    sensors = [
        'position-orientation',
        'linear-velocity-acceleration',
        'accelerometer',
        'angular-velocity',
        'joints-data',
        'toes-contact',
        'thighs-shanks-contact',
        'desired-direction',
        'friction',
        'height-scan'
    ]
    sensors_txt = ''
    for s in sensors[:-1]: sensors_txt += f'"{s}", '
    sensors_txt += f'"{sensors[-1]}"'
    parser.add_argument(
        '-s', '---sensor',
        choices=sensors,
        default='position-orientation',
        help='Sensor to test. Available values: ' + sensors_txt,
        metavar='SENSOR'
    )

    args = parser.parse_args()

    sim = Simulation(args.spot_urdf, gui=True, real_step=args.real_step)
    
    if args.sensor == 'position-orientation': 
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_position_orientation
    elif args.sensor == 'linear-velocity-acceleration':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_linear_velocity_acceleration
    elif args.sensor == 'accelerometer':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_acceleration_free_fall
    elif args.sensor == 'angular-velocity':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_angular_velocity
    elif args.sensor == 'joints-data':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_joint_sensors
    elif args.sensor == 'toes-contact':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_toes_contact
    elif args.sensor == 'thighs-shanks-contact':
        sim.reset('terrains/contact_test.txt')
        test_function = sim.test_thighs_shanks_contact
    elif args.sensor == 'desired-direction':
        sim.reset('terrains/initial_hills.txt')
        test_function = sim.test_desired_direction
    elif args.sensor == 'friction':
        sim.reset('terrains/friction_test.txt')
        test_function = sim.test_friction
    elif args.sensor == 'height-scan':
        sim.reset('terrains/height_scan_test.txt')
        test_function = sim.test_height_scan

    sim.test(test_function)