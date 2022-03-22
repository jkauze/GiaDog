from dataclasses import dataclass
from typing import *

@dataclass
class Joint:
    index  : int
    name   : str
    type   : int
    gIndex : int
    uIndex : int
    flags  : int
    damping     : float
    friction    : float
    lowerLimit  : float
    upperLimit  : float
    maxForce    : float
    maxVelocity : float
    linkName    : str
    axis        : tuple
    parentFramePosition    : tuple
    parentFrameOrientation : tuple
    parentIndex : int

    def __post_init__(self):
        self.name = str(self.name, 'utf-8')
        self.linkName = str(self.linkName, 'utf-8')

@dataclass
class JointState:
    jointPosition           : float
    jointVelocity           : float
    jointReactionForces     : tuple
    appliedJointMotorTorque : float

@dataclass
class SensorsState:
    gyro_angles  : list
    gyro_angular_velocities: list
    joint_angles : list
    toe_states   : list

@dataclass
class ContactInfo:
    contactFlag   : int  # reserved
    bodyUniqueIdA : int  # body unique id of body A
    bodyUniqueIdB : int  # body unique id of body B
    linkIndexA    : int  # link index of body A, -1 for base  
    linkIndexB    : int  # link index of body B, -1 for base
    positionOnA   : list # vec3//list of 3 floats
                         # contact position on A, in Cartesian world coordinates
    positionOnB   : list # vec3, list of 3 floats
                         # contact position on B, in Cartesian world coordinates
    contactNormalOnB    : list  # vec3, list of 3 float// contact normal on B, 
                                # pointing towards A
    contactDistance     : float # contact distance, positive for separation, 
                                # negative for penetration
    normalForce         : float # normal force applied during the last 
                                # 'stepSimulation'
    lateralFriction1    : float # lateral friction force in the 
                                # lateralFrictionDir1 direction
    lateralFrictionDir1 : list  # vec3, list of 3 floats // first lateral 
                                # friction direction
    lateralFriction2    : float # lateral friction force in the 
                                # lateralFrictionDir2 direction
    lateralFrictionDir2 : list  # vec3, list of 3 floats// second lateral 
                                # friction direction

@dataclass
class LinkState:
    linkWorldPosition : List[float] # vec3, list of 3 floats
                                    # Cartesian position of center of mass
    
    linkWorldOrientation : List[float] # vec4, list of 4 floats
                                       # Cartesian orientation of center of mass, 
                                       # in quaternion [x,y,z,w]
    
    localInertialFramePosition:List[float] # vec3, list of 3 floats
                                           # local position offset of inertial frame 
                                           # (center of mass) expressed in the URDF 
                                           # link frame
    localInertialFrameOrientation:List[float] # vec4, list of 4 floats
                                              # local orientation (quaternion [x,y,z,w]) 
                                              # offset of the inertial frame expressed in 
                                              # URDF link frame.
    worldLinkFramePosition:List[float]
    #vec3, list of 3 floats
    #world position of the URDF link frame
    worldLinkFrameOrientation:List[float]
    #vec4, list of 4 floats
    #world orientation of the URDF link frame
    worldLinkLinearVelocity:Optional[List[float]] = None 
    #vec3, list of 3 floats
    #Cartesian world velocity. Only returned if computeLinkVelocity non-zero.
    worldLinkAngularVelocity:Optional[List[float]] = None 
    #vec3, list of 3 floats
    #Cartesian world velocity. Only returned if computeLinkVelocity non-zero.






