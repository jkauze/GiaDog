"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""
import numpy as np
from typing import *

class controller:
    """
        [TODO: DESCRIPTION]
    """
    # This distance are in meters
    
    THIGH_JOINT_LEN =  0.11058
    SHANK_JOINT_LEN  = 0.1265
    # Vertical offset betweet hip and shoulder link.
    VER_OFFSET       = 0.00869998 #0.00869998
    # Horizontal offset betweet hip and shoulder link.
    HOR_OFFSET         =  0.063


    SIGMA_0 = np.array([0,np.pi/2,np.pi,np.pi*1.5])
        
    @classmethod     
    def get_domain(cls, x:float, y:float, z:float) -> float:
        """
        Calculates the leg's Domain and caps it in case of a breach
        :param x,y,z: hip-to-foot distances in each dimension
        :return: Leg Domain D
        """

        """
        Calculates the leg's Domain D and caps it in case of a breach
        
        Arguments:
        ---------
        leg_type: string ('LEFT' or 'RIGHT')
                    Parameter that defines the leg type
        
        x,y,z: -> float hip-to-foot distances in each dimension
        
        Return:
        -------
        D: -> float leg's Domain D
        
        """
        
        h_2 = np.sqrt(z ** 2 + y ** 2)
        sqrt_component = h_2 ** 2 - cls.HOR_OFFSET ** 2
        if sqrt_component < 0:
            sqrt_component = 0
        r_o = np.sqrt(sqrt_component) - cls.VER_OFFSET

        D = (r_o**2 + x**2 - cls.SHANK_JOINT_LEN**2 -cls.THIGH_JOINT_LEN**2)/\
            (2*cls.SHANK_JOINT_LEN*cls.THIGH_JOINT_LEN)
        if D > 1 or D < -1:
            # DOMAIN BREACHED
            #print("---------DOMAIN BREACH---------")
            D = np.clip(D, -1.0, 1.0)
            return D
        else:
            return D
    

    @classmethod
    def solve_leg_IK(cls, leg_type:str, r:np.ndarray) -> np.ndarray:
        """
        Calculates the leg's inverse kinematics.
        (joint angles from xyz coordinates).
        Arguments:
        ---------
        cls: controller class
        leg_type: string ('LEFT' or 'RIGHT')
                    Parameter that defines the leg type
        r : numpy.ndarray, shape (3,)
            Objective foot position in the H_i frame.
            (x,y,z) hip-to-foot distances in each dimension
        
        Return:
        -------
        numpy.ndarray, shape (3,) 
            Leg joint angles to reach the objective foot position r. In the 
            order:(Hip, Shoulder, Wrist). The joint angles are expresed in 
            radians.
        
        References:
        -----------
            * Muhammed Arif Sen, Veli Bakircioglu, Mete Kalyoncu. (Sep, 2017). 
            Inverse Kinematic Analysis Of A Quadruped Robot  
            https://www.researchgate.net/publication/320307716_Inverse_Kinematic_Analysis_Of_A_Quadruped_Robot

            * Some of the code was taken from the sopt_mini_mini implementation 
            of the same paper.
            https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
        """
        x,y,z = r
        
        # Calculate the leg domain
        D = cls.get_domain(x, y, z)
        # Depending on the leg type, calculate the inverse kinematics
        if leg_type == "RIGHT":
            return cls.RightIK(x, y, z, D)
        else:
            return cls.LeftIK(x, y, z, D)

    
    @classmethod
    def RightIK(cls, x:float, y:float, z:float, D:float) -> np.ndarray:
        """
        Right Leg Inverse Kinematics Solver
        
        Arguments:
        ---------
        cls: -> controller class
        x,y,z: -> float hip-to-foot distances in each dimension
        D: -> float leg domain
        
        Return:
        -------
        joint_angles : -> np.ndarray Joint Angles required for desired position.
                        The order is: shoulder, elbow, wrist
                        Or: (Hip, Thigh, Shank)
        """
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
        
        sqrt_component = y**2 + z**2 - cls.HOR_OFFSET**2
        
        if sqrt_component < 0.0:
            #print("NEGATIVE SQRT")
            sqrt_component = 0.0

        r_o = np.sqrt(sqrt_component) - cls.VER_OFFSET

        second_sqrt_component = r_o**2+x**2 - \
                                (cls.SHANK_JOINT_LEN * np.sin(wrist_angle))**2
        if  second_sqrt_component < 0.0:
            print("NEGATIVE SQRT")
            q_o = 0.0
        else:
            q_o = np.sqrt(second_sqrt_component)

        shoulder_angle = -np.arctan2(z, y) - np.arctan2(
            r_o, -cls.HOR_OFFSET)
        
        elbow_angle = np.arctan2(-x, r_o) -\
            np.arctan2(cls.SHANK_JOINT_LEN * np.sin(wrist_angle),
            q_o
            )
            
        
        joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
        
        return joint_angles

    @classmethod
    def LeftIK(cls, x:float, y:float, z:float, D:float) -> np.ndarray:
        """
        Left Leg Inverse Kinematics Solver
        
        Arguments:
        ---------
        cls: -> controller class
        x,y,z: -> float hip-to-foot distances in each dimension
        D: -> float leg domain
        
        Return:
        -------
        joint_angles : -> np.ndarray Joint Angles required for desired position.
                        The order is: shoulder, elbow, wrist
                        Or: (Hip, Thigh, Shank)
        """
        wrist_angle = np.arctan2(-np.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z)**2 - cls.HOR_OFFSET**2
        if sqrt_component < 0.0:
            print("NEGATIVE SQRT")
            sqrt_component = 0.0
        
        r_o = np.sqrt(sqrt_component) - cls.VER_OFFSET

        second_sqrt_component = r_o**2+x**2 - \
                                (cls.SHANK_JOINT_LEN * np.sin(wrist_angle))**2
        if  second_sqrt_component < 0.0:
            print("NEGATIVE SQRT")
            q_o = 0.0
        else:
            q_o = np.sqrt(second_sqrt_component)

        shoulder_angle = -np.arctan2(z, y) - np.arctan2(r_o, 
                                                        cls.HOR_OFFSET)
        
        elbow_angle = np.arctan2(-x, r_o) -\
            np.arctan2(cls.SHANK_JOINT_LEN * np.sin(wrist_angle),
            q_o
            )
        
        joint_angles = np.array([-shoulder_angle, elbow_angle, wrist_angle])
        return joint_angles


    @staticmethod
    def FTG(
            Hi_z: np.ndarray, 
            sigma_i_0: float, 
            t: float, 
            f_i: float,
            h: float = 0.067, # 0.2 is the parameter used in the ETH-Zurich paper 
            f_0 : float= 24 #1.25 is the parameter used in the ETH-Zurich paper
            ) :
        """
        Generates a vector in R^3 representing the desired foot position (end 
        efector) in the H_i frame corresponding to the robots i-th leg 
        horizontal frame below its hip.
        
        Arguments:
        ----------
        Hi_z : np.ndarray, shape(3,)
            i-th leg horizontal frame z component.
        sigma_i_0 : float 
            Contact phase.
        t  : float 
            Timestep.
        f_i: float 
            i-th leg frequency offset (from NN policy).
        h : float, optional
            Maximun foot height in meters. 
            Default 0.2
        f_0 : float, optional
            Robot gait common frequency in Hz. 
            Default 1.25 
            
        Return:
        -------
        Tuple:

        numpy.ndarray, shape (3,) 
            Vector expresed in the i-th leg horizontal frame Hi, representing de 
            target foot position.
        
        sigma_i_0 : float
            FTG frequency. This output is used as an input of the neural network

        References:
        -----------
            * Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
            (p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
            https://arxiv.org/pdf/2010.11251.pdf

            * A detailed explanation of the Horizontal frame con be found on 
            this paper: A Reactive Controller Framework for Quadrupedal
            Locomotion on Challenging Terrain. (Barasuoul, et al., 2013).
            https://iit-dlslab.github.io/papers/barasuol13icra.pdf



        """
        sigma_i = (sigma_i_0 + t * (f_0 + f_i)) % (2 * np.pi)
        k = 2 * (sigma_i - np.pi) / np.pi
        
        param = 0 # 0.5
        if 0 < k < 1: 
            return (h * (-2 * k ** 3 + 3 * k ** 2) - param) * Hi_z, sigma_i
        elif 1 < k < 2: 
            return (h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4) - param) * Hi_z, sigma_i
        else: 
            return - param * Hi_z, sigma_i
    
    @classmethod    
    def apply_FTG(cls, theta, t):
        """
        Processes the NN output theta and outputs the desired foot position in 
        the H_i frame corresponding to the robots i-th leg horizontal frame
        below its hip, the function also outputs the FTG_frequencies and 
        FTG_phases of each leg.

        Arguments:
        ----------
        cls: -> controller class
        theta: -> np.ndarray, shape(16,), output of the NN
        t: -> float, timestep of the simulation. (In seconds)

        Return:
        -------
        Tuple:
        target_foot_psoitions: -> np.ndarray, shape(4,3)
            Desired foot positions in the H_i frame corresponding to the robots
            i-th leg horizontal frame below its hip.
        FTG_frequencies: -> np.ndarray, shape(4,)
            Foot Trajectory Generator frequencies of each leg.
        FTG_phases: -> np.ndarray, shape(4,)
            Foot Trajectory Generator phases of each leg.
        
        """
        
        target_foot_positions = np.zeros([4,3])
        FTG_frequencies = np.zeros([4])
        FTG_phases = np.zeros([4,2])
        
        # Unless gravity changes (which will be very rare), the Hi_z vector is 
        # constant, and parallel to the ground.
        Hi_z = np.array([0,0,1])
        
        for i in range(4):
            xyz_residual = theta[i+0:3+i]
            f_i = theta[i+3]
            sigma_i_0 = cls.SIGMA_0[i]
            r, sigma_i = cls.FTG(Hi_z, sigma_i_0, t, f_i) 

            FTG_frequencies[i] = sigma_i
            FTG_phases[i] = [np.sin(sigma_i), np.cos(sigma_i)]
            target_foot_positions[i] = r + xyz_residual
        
        return target_foot_positions, FTG_frequencies, FTG_phases
    
    @classmethod
    def apply_IK(cls, T, r, leg_type = "RIGHT"):
        """
        Applies the Inverse Kinematics algorithm given a position r in the 
        Horizontal Frame, by apliying the correspondig homogeneous 
        transformation T from H_i to the Hip frame.

        Arguments:
        ----------
        cls: -> controller class
        T: -> np.ndarray, shape(4,4), homogeneous transformation from H_i to 
              the Hip frame.
        r: -> np.ndarray, shape(3,), position in the Horizontal Frame.
        leg_type: -> str, "RIGHT" or "LEFT"

        Return:
        -------
        joint_angles: -> np.ndarray, shape(3,)
            Joint angles of the robot.
        """
        r_hip = T @ np.array([*r, 1]).T
        
        joint_angles = cls.solve_leg_IK(leg_type, r_hip[0:3])
        
        return joint_angles
        
  
            



    
    
    