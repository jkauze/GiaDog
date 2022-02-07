"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
    
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

import numpy as np


def FTG(sigma_i_0: float, 
        t: float, 
        f_i: float,
        Hi_z: np.ndarray = np.array([0,0,1]), 
        h: float = 0.067, # 0.2 is the parameter used in the ETH-Zurich paper 
        f_0 : float= 1.25 #1.25 is the parameter used in the ETH-Zurich paper
        ) -> tuple :
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

    """
    sigma_i = (sigma_i_0 + t * (f_0 + f_i)) % (2 * np.pi)
    k = 2 * (sigma_i - np.pi) / np.pi
    
    param = 0 # 0.5
    
    if 0 < k < 1: 
        return (h * (-2 * k ** 3 + 3 * k ** 2) - param) * Hi_z, sigma_i
    elif 1 < k < 2: 
        return (h * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4) - param) * Hi_z,\
                 sigma_i
    else: 
        return - param * Hi_z, sigma_i


def calculate_foot_trajectories(
        theta: np.ndarray,
        t: float, 
        h: float = 0.067, # 0.2 is the parameter used in the ETH-Zurich paper 
        f_0 : float= 1.25, #1.25 is the parameter used in the ETH-Zurich paper
        sigma_0:np.ndarray = np.array([0,np.pi/2,np.pi,1.5 *np.pi]),
        Hi_z: np.ndarray = np.array([0,0,1])
        ) -> tuple:
    """
    Processes the NN output theta and outputs the desired foot position in 
    the H_i frame corresponding to the robots i-th leg horizontal frame
    below its hip, the function also outputs the FTG_frequencies and 
    FTG_phases of each leg.

    Arguments:
    ----------
    theta: -> np.ndarray, shape(16,)
            Output of the NN
    
    t: -> float, 
        Timestep of the simulation. (In seconds)

    h :-> float. Optional.
        Maximun foot height in meters. 
        Default 0.2
    
    f_0 :-> float. Optional.
        Robot gait common frequency in Hz. 
        Default 1.25
    
    sigma_0 :-> np.ndarray, shape(4,). Optional.
        Initial foots phases. Values are in radians between 0 and 2pi.
    
    Hi_z :-> np.ndarray, shape(3,). Optional.
        i-th leg horizontal frame z component. Must be parallel to the 
        gravity vector.


    Return:
    -------
    Tuple:
    
    target_foot_psoitions: -> np.ndarray, shape(4,3)
        Desired foot positions in the H_i frame corresponding to the robots i-th 
        leg horizontal frame below its hip.
    
    FTG_frequencies: -> np.ndarray, shape(4,)
        Foot Trajectory Generator frequencies of each leg.
    
    FTG_phases: -> np.ndarray, shape(4,)
        Foot Trajectory Generator phases of each leg.
    
    """
    
        
    target_foot_positions = np.zeros([4,3])
    FTG_frequencies = np.zeros([4])
    FTG_phases = np.zeros([4,2])
    

    for i in range(4):
        xyz_residual = theta[i : i+3]
        f_i = theta[i + 3]
        sigma_i_0 = sigma_0[i]
        r, sigma_i = FTG(sigma_i_0, 
                         t, 
                         f_i,
                         Hi_z = Hi_z,
                         h = h, 
                         f_0 = f_0
                        )

        FTG_frequencies[i] = sigma_i
        FTG_phases[i] = [np.sin(sigma_i), np.cos(sigma_i)]
        target_foot_positions[i] = r + xyz_residual
    
    return target_foot_positions, FTG_frequencies, FTG_phases
    