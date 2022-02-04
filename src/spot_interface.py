#!/usr/bin/env python3
"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO]
"""
# Utilities
from typing import *

# Simulation
import rospy
import message_filters
from spot_mini_ros.msg import joint_angles, normal_data, priviliged_data

# Training
from src.policy import *
from src.controller import *


class spot_interface:
    """
        [TODO]
    """
    JOINTS_QUEUE_SIZE = 10
    DATA_QUEUE_SIZE = 10
    # ROS publisher node that update the spot mini joints
    joints_pub = rospy.Publisher('spot_joints', joint_angles, JOINTS_QUEUE_SIZE)

    def __init__(self, function: Callable, priviliged: bool=True):
        # ROS subscriber nodes that gets the simulation data
        self.normal_data_sub = rospy.Subscriber('normal_data', normal_data)

        if priviliged:
            self.priviliged_data_sub = rospy.Subscriber(
                'priviliged_data', 
                priviliged_data
            )
            self.ts = message_filters.TimeSynchronizer(
                [self.normal_data_sub, self.priviliged_data_sub], 
                self.DATA_QUEUE_SIZE
            )
        else:
            self.ts = message_filters.TimeSynchronizer(
                [self.normal_data_sub], 
                self.DATA_QUEUE_SIZE
            )

        self.ts.registerCallback(function)

    @classmethod
    def actuate_joints(cls, joint_target_positions: List[float]):
        """
            Moves the robot joints to a given target position.

            Argument:
            ---------
                joint_target_positions: List[float], shape (12,)
                    Quadruped joints desired angles. 
                    The order is the same as for the robot actuated_joints_ids.
                    The order should be as follows:
                        'motor_front_left_hip' 
                        'motor_front_left_upper_leg'// "Front left thigh"
                        'motor_front_left_lower_leg'// "Front left shank"

                        'motor_front_right_hip' 
                        'motor_front_right_upper_leg'// "Front right thigh"
                        'motor_front_right_lower_leg'// "Front right shank"

                        'motor_back_left_hip' 
                        'motor_back_left_upper_leg'// "Back left thigh"
                        'motor_back_left_lower_leg'// "Back left shank"

                        'motor_back_right_hip' 
                        'motor_back_right_upper_leg'// "Back right thigh"
                        'motor_back_right_lower_leg'// "Back right shank"
        """
        # Create message
        msg = joint_angles()
        msg.joint_target_positions = joint_target_positions
        cls.joints_pub.publish(msg)

if __name__ == '__main__':
    # Create and initialize ROS node
    spot_interface(lambda x, y : print(x, '\n', y))
    rospy.init_node('spot_interface', anonymous=True)

    # The node keeps runing until the process is canceled
    rospy.spin()
