"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    [TODO: DESCRIPTION]
"""
import numpy as np


def euler_angles_to_rotation_matrix(theta) :
    """
    Calculates a rotation matrix from the euler angles.

    Arguments:
    ----------
    theta:-> np.ndarray, shape(3,)
        Euler angles. (roll, pitch, yaw)
    
    Return:
    -------
    R:- >np.ndarray, shape(3,3)
        Rotation matrix.
    """
    
    roll, pitch, yaw = theta[0], theta[1], theta[2]

    R_x = np.array([[1,         0,                  0        ],
                    [0,         np.cos(roll), -np.sin(roll)  ],
                    [0,         np.sin(roll),  np.cos(roll)  ]
                    ])

    R_y = np.array([[np.cos(pitch),    0,      np.sin(pitch)],
                    [0,                1,         0         ],
                    [-np.sin(pitch),   0,      np.cos(pitch)]
                    ])

    R_z = np.array([[np.cos(yaw),    -np.sin(yaw),     0],
                    [np.sin(yaw),     np.cos(yaw),     0],
                    [0,                     0,         1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def get_leg_to_horizontal_frame_transformations(base_rpy:np.ndarray,
                                               leg_span:float = 0.2442,
                                               leg_offset:float = 0.063,):
    """
        Returns the transformation matrices from the hip to the leg base.
    
    Arguments:
    ---------
    base_rpy: -> np.ndarray, shape(3,) 
        The hip's (and robot base) euler angles. (roll, pitch, yaw)
    
    robot_width: -> float
        The robot's width (in meters) from hip to hip (left hip to right hip).
    
    robot_lenght: -> float
        The robot's lenght (in meters) from hip to hip (front hip to rear hip).

    
    leg_span: - >float
        Distance of the extended leg.

    Returns:
    --------
    Transformation_matrices: List[np.array], shape (4,4). A list containing 
        the transformation matrices from the hip to the leg base, for each 
        of the robots legs.

    """
    Transformation_matrices = []
    
    # We transform the base orientation from quaternion to matrix
    # We also get the base euler angles
    base_roll, base_pitch, base_yaw = base_rpy

    for i in range(4):
        # We calculate the Hi frame  relative to the leg base position (hip)
        p_li_hi = np.array([0,  leg_offset * (-1)**i,  -leg_span])
        # We do the same for the leg base orientation
        R_li_Hi = euler_angles_to_rotation_matrix([base_roll, base_pitch, 0])

        # Finally we concatenate the rotation matrix and position vector
        # To get the transformation matrix of the Hi horizontal frame expressed
        # in the leg base frame
        T_li_Hi =  np.concatenate((R_li_Hi,  p_li_hi.reshape(3,1)), axis = 1)
        T_li_Hi = np.concatenate((T_li_Hi, np.array([[0,0,0,1]])), axis = 0)

        Transformation_matrices.append(T_li_Hi)
    
    return Transformation_matrices
