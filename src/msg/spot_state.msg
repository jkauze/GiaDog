# Values that can be obtained from the simulator for agent training

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

# =========== Privileged Data =========== #
float32[] joint_torques

# Normal at each toe
float32[] normal_toe_fl  # Front left 
float32[] normal_toe_fr  # Front right 
float32[] normal_toe_bl  # Back left 
float32[] normal_toe_br  # Back right 

# Force at each toe 
float32[] toes_force1 
float32[] toes_force2 

# Ground friction coefficients at each toe 
float32[] ground_friction 

# Height scan at each toe 
float32[] height_scan_fl  # Front left
float32[] height_scan_fr  # Front right
float32[] height_scan_bl  # Back left
float32[] height_scan_br  # Back right
