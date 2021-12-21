"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""
import numpy as np
from typing import *

class teacher_policy:
    """
        [TODO: DESCRIPTION]
    """
    VEL_TH   = 0.6 # Velocity threshold
    SWIGN_PH = 0   # Swign phase

    @classmethod
    def reward(
        cls,
        target_dir: np.ndarray,
        turn_dir: int,
        linear_vel: np.ndarray,
        angular_vel: np.ndarray,
        ftg_freqs: List[float],
        height_scan: List[np.ndarray],
        thigh_contact: List[bool],
        shank_contact: List[bool],
        foot_target_hist: List[List[np.ndarray]],
        joints_pos: List[float]
        ) -> float:
        """
            Reward function.

            Arguments:
            ----------
            target_dir: np.ndarray, shape (2,)
                Target horizontal direction.
            turn_dir: int
                Turning direction.
            linear_vel: np.ndarray, shape (2,)
                Base linear velocity.
            angular_vel: np.ndarray, shape (3,)
                Base angular velocity.
            ftg_freqs: List[float], len 4
                FTG frequencies.
            height_scan: np.ndarray, shape (4, 12)
                Height scan around each foot.
            thigh_contact: List[bool], len 4
                Thigh contact states.
            shank_contact: List[bool], len 4
                Shank contact states.
            foot_target_hist: np.ndarray, shape (3, 4, 3)
                Foot target history at times t, t-1 and t-2.
            joints_pos: List[float], len
                Joints positions

            Return:
            -------
            float 
                Reward value.

            References:
            -----------
                * 	Learning Quadrupedal Locomotion over Challenging Terrain (Oct,2020).
                    (p.8 Motion synthesis and p.15 S3 Foot trajectory generator).
                    https://arxiv.org/pdf/2010.11251.pdf
        """
        # Zero command
        zero = not (target_dir[0] or target_dir[1])

        # Base horizontal linear velocity projected onto the command direction.
        proj_linear_vel = np.dot(linear_vel, target_dir)
        # Velocity orthogonal to the target direction.
        ort_vel = (linear_vel - proj_lineal_vel * target_dir) if zero else linear_vel
        ort_vel = numpy.linalg.norm(ort_vel)

        # Base horizontal angular velocity.
        hor_angular_vel = np.ndarray([angular_vel[0], angular_vel[1]])
        # Base angular velocity Z projected onto desired angular velocity.
        proj_angular_vel = angular_vel[2] * turn_dir

        # Set of such collision-free feet and index set of swing legs
        count_swing = 0
        foot_clear = 4
        for i in range(4):
            # If i-th foot is in swign phase.
            if ftg_freqs[i] >= cls.SWIGN_PH:
                count_swign += 1

                # Verify that the height of the i-th foot is greater than the height of 
                # the surrounding terrain
                foot_clear_i = True
                for height in height_scan:
                    if foot_target_hist[0][i][2] <= height:
                        foot_clear -= 1
                        break

        # ======================= REWARDS ======================= #
        # Linear Velocity Reward
        if zero:
            r_lv = 0
        elif proj_linear_vel < cls.VEL_TH:
            r_lv = np.exp(-2 * (proj_linear_vel - cls.VEL_TH) ** 2)
        else:
            r_lv = 1

        # Angular Velocity Reward
        double r_av = 0
        if turn_dir == 0:
            r_av = 0
        elif proj_angular_vel < cls.VEL_TH:
            r_av = np.exp(-1.5 * (proj_angular_vel - cls.VEL_TH) ** 2)
        else:
            r_av = 1

        # Base Motion Reward
        w_2 = np.dot(h_angular_vel, h_angular_vel)
        r_b = np.exp(-1.5 * ort_vel ** 2) + np.exp(-1.5 * w_2)

        # Foot Clearance Reward
        r_fc = foot_clear / count_swing if count_swign > 0 else 1

        # Body Collision Reward
        r_bc = -sum(thigh_contact) - sum(shank_contact)

        # Target Smoothness Reward
        r_fd_T = []
        for i in range(3):
            r_fd_T.append(np.ndarray([
                foot_target_hist[i][0][0], foot_target_hist[i][0][1], 
                foot_target_hist[i][0][2], foot_target_hist[i][1][0], 
                foot_target_hist[i][1][1], foot_target_hist[i][1][2],
                foot_target_hist[i][2][0], foot_target_hist[i][2][1], 
                foot_target_hist[i][2][2], foot_target_hist[i][3][0], 
                foot_target_hist[i][3][1], foot_target_hist[i][3][2]
            ]))
        r_s = -np.linalg.norm(r_fd_T[0] - 2.0 * r_fd_T[1] + r_fd_T[2])

        # Torque Reward
        r_tau = 0
        for pos in joints_pos: r_tau -= abs(pos)

        return (5*r_lv + 5*r_av + 4*r_b + r_fc + 2*r_bc + 2.5*r_s) / 100.0 + 2e-5 * r_tau