"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""
import numpy as np

class controller:
    """
        [TODO: DESCRIPTION]
    """
    SHOULDER_JOINT_LEN = 1
    WRIST_JOINT_LEN    = 1
    VER_OFFSET         = 1  # Vertical offset betweet hip and shoulder link.
    HOR_OFFSET         = 1  # Horizontal offset betweet hip and shoulder link.

    @staticmethod
    def FTG(
            Hi_z: np.ndarray, 
            sigma_i_0: float, 
            t: float, 
            f_i: float,
            h: float=0.2,
            f_0 : float=1.25
            ) -> np.ndarray:
        """
            Generates a vector in R^3 representing the desired foot position (end efector)
            in the H_i frame corresponding to the robots i-th leg horizontal frame below 
            its hip.
            
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
            numpy.ndarray, shape (3,) 
                Vector expresed in the i-th leg horizontal frame Hi, representing de 
                target foot position.

            References:
            -----------
                * 	Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
                    (p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
                    https://arxiv.org/pdf/2010.11251.pdf
        """
        sigma_i = (sigma_i_0 + t * (f_0 + f_i)) % (2 * np.pi)
        k = 2 * (sigma_i - np.pi) / np.pi

        if 0 < k < 1: 
            return (h * (-2 * k ** 3 + 3 * k ** 2) - 0.5) * Hi_z
        elif 1 < k < 2: 
            return (h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4) - 0.5) * Hi_z
        else: 
            return - 0.5 * Hi_z

    @classmethod
    def inverse_kinematics(cls, r: np.ndarray):
        """
            For a robot foot, given a position r in its horizontal frame, the joint angles
            are calcultated. (Hip, Shoulder, Wrist) [The angles are in radians]
            
            Arguments:
            ---------
            r : numpy.ndarray, shape (3,)
                Objective foot position in the H_i frame.
                
            Return:
            -------
            numpy.ndarray, shape (3,) 
                Leg joint angles to reach the objective foot position r. In the order: 
                (Hip, Shoulder, Wrist). The joint angles are expresed in radians.
                    
            References:
            -----------
                *   Adham Elarabawy (May, 2020). 12-DOF Quadrupedal Robot: Inverse 
                    Kinematics. https://www.adham-e.dev/pdf/IK_Model.pdf
        """
        x = r[0], y = r[1], z = r[2]

        # Auxiliary Variables
        h_2 = np.sqrt(z ** 2 + y ** 2)
        r_o = np.sqrt(h_2 ** 2 - cls.HOR_OFFSET ** 2) - cls.VER_OFFSET
        h = np.sqrt(r_o ** 2 + x ** 2)

        # Joint angles calculation
        theta_h = np.arctan(y / z) - np.arcsin(cls.HOR_OFFSET / h_2)
        theta_s = h ** 2 + cls.SHOULDER_JOINT_LEN ** 2 - cls.WRIST_JOINT_LEN ** 2
        theta_s = np.arccos(theta_s / (2 * h * cls.SHOULDER_JOINT_LEN)) - np.arcsin(x / h)
        theta_w = cls.WRIST_JOINT_LEN ** 2 + cls.SHOULDER_JOINT_LEN ** 2 - h ** 2
        theta_w = np.arccos(theta_w / (2 * cls.WRIST_JOINT_LEN * cls.SHOULDER_JOINT_LEN))

        return np.ndarray([theta_h, theta_s, theta_w])