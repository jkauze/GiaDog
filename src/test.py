from re import X
from src.simulation import simulation
from src.distributions import make_dist
from src.giadog_gym import teacher_giadog_env 
from src.policy_networks import teacher_network
import pybullet as p
import numpy as np

if __name__ == '__main__':
    # Create simulation
    spot_urdf_file = 'src/mini_ros/urdf/spot.urdf'
    sim = simulation(spot_urdf_file, p,
                self_collision_enabled=False)

    
    env = teacher_giadog_env(sim = sim)
    print("# # # Env Initialized # # #")
    ac_space = env.action_space
    obs_space = env.observation_space
    policy = teacher_network(action_space = ac_space,
                        observation_space = obs_space)
    print("# # # Policy Initialized # # #")

    p_info = np.zeros((1,59))
    np_info = np.zeros((1,145))
    result = policy([p_info, np_info], greedy=True)
    print(result)
    