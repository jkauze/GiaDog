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
from typing import *
from __env__ import F_0, H, H_Z, SIGMA_0

def FTG(sigma_i_0: float, t: float, f_i: float) -> Tuple[np.array, float]:
    """
        Generates a vector in R^3 representing the desired foot position (end 
        efector) in the H_i frame corresponding to the robots i-th leg 
        horizontal frame below its hip.
        
        Arguments:
        ----------
            sigma_i_0 : float 
                Contact phase.

            t  : float 
                Timestep.

            f_i: float 
                i-th leg frequency offset (from NN policy).
            
        Returns:
        -------
            numpy.ndarray, shape (3,) 
                Vector expresed in the i-th leg horizontal frame Hi,
                representing the target foot position.

            float
                FTG frequency. This output is used as an input of the neural 
                network
    """

    sigma_i = (sigma_i_0 + t * (F_0 + f_i)) % (2 * np.pi)
    k = 2 * (sigma_i - np.pi) / np.pi
    param = 0 # 0.5
    
    if 0 < k < 1: 
        position = (H * (-2 * k ** 3 + 3 * k ** 2) - param) * H_Z
    elif 1 < k < 2: 
        position = (H * (2 * k ** 3 - 9 * k ** 2 + 12 * k - 4) - param) * H_Z
    else: 
        position = - param * H_Z

    return position, sigma_i

def foot_trajectories(
        theta: np.array, 
        t: float
    ) -> Tuple[np.array, np.array, np.array]:
    """
        Processes the NN output theta and outputs the desired foot position in 
        the H_i frame corresponding to the robots i-th leg horizontal frame
        below its hip, the function also outputs the FTG_frequencies and 
        FTG_phases of each leg.

        Arguments:
        ----------
            theta: numpy.array, shape(16,)
                Output of the NN

            t: float
                Timestep of the simulation (in seconds).

        Returns:
        -------
            numpy.array, shape(4,3)
                Desired foot positions in the H_i frame corresponding to the 
                robots i-th leg horizontal frame below its hip.
            
            numpy.array, shape(4,)
                Foot Trajectory Generator frequencies of each leg.
            
            numpy.array, shape(4,)
                Foot Trajectory Generator phases of each leg.
    """
    target_foot_positions = np.zeros([4,3])
    FTG_frequencies = np.zeros([4])
    FTG_phases = np.zeros([4,2])
    
    for i in range(4):
        xyz_residual = theta[i : i+3]
        f_i = theta[i + 3]
        sigma_i_0 = SIGMA_0[i]
        r, sigma_i = FTG(sigma_i_0, t, f_i)

        FTG_frequencies[i] = sigma_i
        FTG_phases[i] = [np.sin(sigma_i), np.cos(sigma_i)]
        target_foot_positions[i] = r + xyz_residual
    
    return target_foot_positions, FTG_frequencies, FTG_phases
    
