from dataclasses import dataclass

@dataclass
class Joint:
	index: int
	name: str
	type: int
	gIndex: int
	uIndex: int
	flags: int
	damping: float
	friction: float
	lowerLimit: float
	upperLimit: float
	maxForce: float
	maxVelocity: float
	linkName: str
	axis: tuple
	parentFramePosition: tuple
	parentFrameOrientation: tuple
	parentIndex: int

	def __post_init__(self):
		self.name = str(self.name, 'utf-8')
		self.linkName = str(self.linkName, 'utf-8')

@dataclass
class JointState:
	jointPosition: float
	jointVelocity: float
	jointReactionForces: tuple
	appliedJointMotorTorque: float


@dataclass
class sensors_state:
	gyro_angles: list
	gyro_angular_velocities: list
	joint_angles: list
	toe_states: list



