# Position and orientation
float32[] position 
float32[] orientation

# Velocities
float32[] linear_vel
float32[] angular_vel

# Contact states
bool[] toes_contact
bool[] thighs_contact
bool[] shanks_contact

# Joints states
float32[] joint_angles
float32[] joint_velocities

# Transformation matrices
float32[] transf_matrix

float32[] foot_target
bool is_fallen
