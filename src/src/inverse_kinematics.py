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


def get_IK_params(x:float, y:float, z:float, 
                h_off:float = 0.063,
                v_off:float = 0.008,
                thigh_l:float = 0.11058,
                shank_l:float = 0.1265 ):
    """
    Calculates the leg's Inverse kinematicks parameters:
        The leg Domain 'D' (caps it in case of a breach)
        And the leg's radius.
    
    
    Arguments:
    ---------
    leg_type:-> string 
            ('LEFT' or 'RIGHT') Parameter that defines the leg type.
    
    x:-> float  
        hip-to-foot distance in x-axis
    y:-> float  
        hip-to-foot distance in y-axis
    z:-> float  
        hip-to-foot distance in z-axis
    
    # Leg parameters
    h_off:-> float. Optional.
        Hip horizontal offset
    
    v_off:-> float. Optional.
        Hip vertical offset
    
    thigh_l:-> float. Optional.
        Thigh length
    
    shank_l:-> float. Optional.
        Shank length

    
    Return:
    -------
    D  : -> float.
            leg's Domain D
    r_o: -> float.
            leg's outer radius
    """
    sqrt_component = z ** 2 + y ** 2 - h_off ** 2
    
    if sqrt_component < 0:
        sqrt_component = 0
    
    r_o = np.sqrt(sqrt_component) - v_off

    D = (r_o**2 + x**2 - shank_l**2 -thigh_l**2)/\
        (2*shank_l*thigh_l)
    
    if D > 1 or D < -1:
        # DOMAIN BREACHED
        D = np.clip(D, -1.0, 1.0)
        
        return D, r_o
    
    else:
        return D, r_o



def RightIK(x:float, y:float, z:float, 
            D:float, r_o:float,
            h_off:float = 0.063,
            shank_l:float = 0.1265) -> np.ndarray:
    """
    Right Leg Inverse Kinematics Solver
    
    Arguments:
    ---------
    x:-> float  
        hip-to-foot distance in x-axis
    y:-> float  
        hip-to-foot distance in y-axis
    z:-> float  
        hip-to-foot distance in z-axis
    
    D:-> float
        Leg domain
    
    r_o:-> float
        Radius of the leg
    
    # Leg parameters
    h_off:-> float. Optional.
        Hip horizontal offset
    
    shank_l:-> float. Optional.
        Shank length

    Return:
    -------
    joint_angles : -> np.ndarray shape(3,) 
        Joint Angles required for desired position. 
        The order is: Hip, Thigh, Shank
        Or: (shoulder, elbow, wrist)
    """
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    
    shoulder_angle = - np.arctan2(z, y) - np.arctan2(r_o, - h_off)

    second_sqrt_component = r_o**2 + x**2 - (shank_l * np.sin(wrist_angle))**2
    
    if  second_sqrt_component < 0.0:
        # NEGATIVE SQUARE ROOT
        second_sqrt_component = 0.0
    
    q_o = np.sqrt(second_sqrt_component)

    
    elbow_angle = np.arctan2(-x, r_o) - \
                    np.arctan2(shank_l * np.sin(wrist_angle), q_o)
        
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    
    return joint_angles


def LeftIK(x:float, y:float, z:float, D:float, r_o:float,
            h_off:float = 0.063,
            shank_l:float = 0.1265) -> np.ndarray:
    """
    Left Leg Inverse Kinematics Solver
    
    Arguments:
    ---------
    x:-> float  
        hip-to-foot distance in x-axis
    y:-> float  
        hip-to-foot distance in y-axis
    z:-> float  
        hip-to-foot distance in z-axis
    
    D:-> float
        Leg domain
    
    r_o:-> float
        Radius of the leg
    
    # Leg parameters
    h_off:-> float. Optional.
        Hip horizontal offset
    
    shank_l:-> float. Optional.
        Shank length

    Return:
    -------
    joint_angles : -> np.ndarray shape(3,) 
        Joint Angles required for desired position. 
        The order is: Hip, Thigh, Shank
        Or: (shoulder, elbow, wrist)
    """
    wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
    
    shoulder_angle = - np.arctan2(z, y) - np.arctan2(r_o, h_off)

    second_sqrt_component = r_o**2 + x**2 - (shank_l * np.sin(wrist_angle))**2
    
    if  second_sqrt_component < 0.0:
        # "NEGATIVE SQRT"
        second_sqrt_component = 0.0
    
    q_o = np.sqrt(second_sqrt_component)

    elbow_angle = np.arctan2(-x, r_o) - \
                    np.arctan2(shank_l * np.sin(wrist_angle), q_o)
    
    joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
    
    return joint_angles


def solve_leg_IK(leg_type:str, r:np.ndarray,
                h_off:float = 0.063,
                v_off:float = 0.008,
                thigh_l:float = 0.11058,
                shank_l:float = 0.1265 ) -> np.ndarray:
    """
    Calculates the leg's inverse kinematics.
    (joint angles from xyz coordinates).
    
    Arguments:
    ---------
    
    leg_type:-> string 
            ('LEFT' or 'RIGHT') 
            Parameter that defines the leg type
    
    r : -> numpy.ndarray, shape (3,)
        Objective foot position in the H_i frame.
        (x,y,z) hip-to-foot distances in each dimension
    
    # Leg parameters
    h_off:-> float. Optional.
        Hip horizontal offset
    
    v_off:-> float. Optional.
        Hip vertical offset
    
    thigh_l:-> float. Optional.
        Thigh length
    
    shank_l:-> float. Optional.
        Shank length
    
    Return:
    -------
    numpy.ndarray, shape (3,) 
        Leg joint angles to reach the objective foot position r. In the 
        order:(Hip, Shoulder, Wrist). The joint angles are expresed in 
        radians.
    
    """
    x,y,z = r
    
    # Calculate the leg domain
    D, r_o = get_IK_params(x, y, z, 
                            h_off = h_off, 
                            v_off = v_off, 
                            thigh_l = thigh_l, 
                            shank_l = shank_l)
    # Depending on the leg type, calculate the inverse kinematics
    
    if leg_type == "RIGHT":
        return RightIK(x, y, z, D, r_o,
                        h_off = h_off,
                        shank_l = shank_l)
    else:
        return LeftIK(x, y, z, D, r_o,
                      h_off = h_off,
                      shank_l = shank_l)
