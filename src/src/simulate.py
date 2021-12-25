#!/usr/bin/env python3
"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains the ROS node in charge of publishing and modifying the status of 
    the spot-mini through ROS topics.
"""
import rospy
import threading
import pybullet as p
from sys import argv
from time import sleep
from simulation import simulation
from spot_mini_ros.msg import spot_state, priviliged_spot_state, joint_angles

# Rate messages for second
RATE = 10

def run_simulation(sim: simulation):
    """ Run the simulation and keep updated the values received by each robot sensor. """
    while True: 
        sim.p.stepSimulation()
        sim.update_sensor_output()

def spot_state_publisher(sim: simulation):
    """
        Function in charge of publishing through a ROS topic the non-privileged 
        information (that would be obtained through real sensors) of the spot-mini.
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
            msg.linear_vel       = list(sim.base_linear_velocity)
            msg.angular_vel      = list(sim.base_angular_velocity)
            msg.toes_contact     = list(sim.toes_contact_states)
            msg.thighs_contact   = list(sim.thighs_contact_states)
            msg.shanks_contact   = list(sim.shanks_contact_states)
            msg.joint_angles     = list(sim.joint_angles)
            msg.joint_velocities = list(sim.joint_velocities)

            # Publish
            pub.publish(msg)
            rate.sleep()

    except rospy.ROSInterruptException:
        print('[w] Topic "spot_state" was stopped.')

def priviliged_spot_state_publisher(sim: simulation):
    """
        Function in charge of publishing through a ROS topic the privileged information 
        (which will not be obtained through real sensors, but only in the simulation) of 
        the spot-mini.
    """
    try:
        # Init node.
        pub = rospy.Publisher(
            'priviliged_spot_state', 
            priviliged_spot_state, 
            queue_size=RATE
        )

        # Run until manually stopped
        print('[i] Topic "priviliged_spot_state" is running!')
        rate = rospy.Rate(RATE) 
        while not rospy.is_shutdown():
            # Create message
            msg = priviliged_spot_state()
            msg.normal_toe_fl   = list(sim.terrain_normal_at_each_toe[0])
            msg.normal_toe_fr   = list(sim.terrain_normal_at_each_toe[1])
            msg.normal_toe_bl   = list(sim.terrain_normal_at_each_toe[2])
            msg.normal_toe_br   = list(sim.terrain_normal_at_each_toe[3])
            msg.toes_force1     = list(sim.contact_force_at_each_toe)
            msg.toes_force2     = list(sim.toe_force_sensor)
            msg.ground_friction = list(sim.foot_ground_friction_coefficients)
            msg.height_scan_fl  = list(sim.height_scan_at_each_toe[0])
            msg.height_scan_fr  = list(sim.height_scan_at_each_toe[1])
            msg.height_scan_bl  = list(sim.height_scan_at_each_toe[2])
            msg.height_scan_br  = list(sim.height_scan_at_each_toe[3])
            msg.joint_torques   = list(sim.joint_torques)

            # Publish
            pub.publish(msg)
            rate.sleep()
    except rospy.ROSInterruptException:
        print('[w] Topic "priviliged_spot_state" was stopped.')
    
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


if __name__ == '__main__':
    # Initialize simulation
    print('[i] Starting simulation')
    sim = simulation(terrain_file=argv[1], giadog_urdf_file=argv[2], bullet_server=p)
    sim.initialize() 

    # Run rosnode for spot-mini
    rospy.init_node('spot_state_node', anonymous=True)
    th1 = threading.Thread(target=run_simulation, args=(sim,))
    th2 = threading.Thread(target=spot_state_publisher, args=(sim,))
    th3 = threading.Thread(target=priviliged_spot_state_publisher, args=(sim,))
    th4 = threading.Thread(target=joint_angles_setter, args=(sim,))
    th1.start()
    th2.start()
    th3.start()
    th4.start()
