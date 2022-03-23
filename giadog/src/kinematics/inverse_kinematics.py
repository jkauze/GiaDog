"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
    
    References:
    -----------
        * Muhammed Arif Sen, Veli Bakircioglu, Mete Kalyoncu. (Sep, 2017). 
        Inverse Kinematic Analysis Of A Quadruped Robot  
        https://www.researchgate.net/publication/320307716_Inverse_Kinematic_Analysis_Of_A_Quadruped_Robot

        * Some of the code was taken from the sopt_mini_mini implementation 
        of the same paper.
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
"""
import numpy as np
from typing import *
from src.kinematics.__env__ import H_OFF, V_OFF, SHANK_L, THIGH_L


def __get_IK_params(x: float, y: float, z: float) -> Tuple[float, float]:
    """
        Calculates the leg's Inverse kinematicks parameters:
        The leg Domain 'D' (caps it in case of a breach) and the leg's radius.

        Arguments:
        ----------
            x: float  
                hip-to-foot distance in x-axis

            y: float  
                hip-to-foot distance in y-axis

            z: float  
                hip-to-foot distance in z-axis

        Returns:
        -------
            float
                leg's Domain D

            float
                leg's outer radius
    """
    sqrt_component = max(0, z ** 2 + y ** 2 - H_OFF ** 2)
    r_o = np.sqrt(sqrt_component) - V_OFF
    D = np.clip(
        (r_o**2 + x**2 - SHANK_L**2 - THIGH_L**2) / (2 * SHANK_L * THIGH_L),
        -1.0, 
        1.0
    )
    
    return D, r_o

def __right_IK(x: float, y: float, z: float, D: float, r_o: float) -> np.array:
    """
        Right Leg Inverse Kinematics Solver
        
        Arguments:
        ---------_
            x: float  
                hip-to-foot distance in x-axis

            y: float  
                hip-to-foot distance in y-axis

            z: float  
                hip-to-foot distance in z-axis
            
            D: float
                Leg domain
            
            r_o: float
                Radius of the leg

        Return:
        -------
            numpy.array, shape(3,) 
                Joint Angles required for desired position. 
                The order is: Hip, Thigh, Shank
                Or: (shoulder, elbow, wrist)
    """
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    shoulder_angle = - np.arctan2(z, y) - np.arctan2(r_o, - H_OFF)
    second_sqrt_component = max(
        0,
        r_o**2 + x**2 - (SHANK_L * np.sin(wrist_angle))**2
    )
    q_o = np.sqrt(second_sqrt_component)
    elbow_angle = np.arctan2(-x, r_o) 
    elbow_angle -= np.arctan2(SHANK_L * np.sin(wrist_angle), q_o)
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    
    return joint_angles

def __left_IK(x: float, y: float, z: float, D: float, r_o: float) -> np.array:
    """
        Left Leg Inverse Kinematics Solver
        
        Arguments:
        ---------
            x: float  
                hip-to-foot distance in x-axis

            y: float  
                hip-to-foot distance in y-axis
                
            z: float  
                hip-to-foot distance in z-axis
            
            D: float
                Leg domain
            
            r_o: float
                Radius of the leg

        Return:
        -------
            np.array, shape(3,) 
                Joint Angles required for desired position. 
                The order is: Hip, Thigh, Shank
                Or: (shoulder, elbow, wrist)
    """
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    shoulder_angle = - np.arctan2(z, y) - np.arctan2(r_o, H_OFF)
    second_sqrt_component = max(
        0,
        r_o**2 + x**2 - (SHANK_L * np.sin(wrist_angle))**2
    )
    q_o = np.sqrt(second_sqrt_component)
    elbow_angle = np.arctan2(-x, r_o) 
    elbow_angle -= np.arctan2(SHANK_L * np.sin(wrist_angle), q_o)
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    
    return joint_angles

def solve_leg_IK(leg_type: str, r: np.array) -> np.array:
    """
        Calculates the leg's inverse kinematics.
        (joint angles from xyz coordinates).
        
        Arguments:
        ---------_
            leg_type: string 
                    ('LEFT' or 'RIGHT') 
                    Parameter that defines the leg type
            
            r: numpy.ndarray, shape (3,)
                Objective foot position in the H_i frame.
                (x,y,z) hip-to-foot distances in each dimension

        Return:
        -------
            numpy.ndarray, shape (3,) 
                Leg joint angles to reach the objective foot position r. In the 
                order:(Hip, Shoulder, Wrist). The joint angles are expresed in 
                radians.
    """
    x, y, z = r
    
    # Calculate the leg domain
    D, r_o = __get_IK_params(x, y, z)

    # Depending on the leg type, calculate the inverse kinematics
    if leg_type == "RIGHT": return __right_IK(x, y, z, D, r_o)
    else: return __left_IK(x, y, z, D, r_o)

