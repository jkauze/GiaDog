from re import X
from src.simulation import simulation
from src.giadog_gym import teacher_giadog_env 
from src.policy_networks import teacher_network
from src.value_networks import teacher_value_network
from src.PPO import PPO_CLIP
import pybullet as p
import numpy as np
from tensorflow.keras.optimizers import Adam


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
    value_function = teacher_value_network()
    print("# # # Policy Initialized # # #")
    policy_optimizer = Adam(lr=0.001)
    value_function_optimizer = Adam(lr=0.001)
    print("# # # Optimizers Initialized # # #")
    net_list = [value_function, policy]
    optimizers_list = [value_function_optimizer ,policy_optimizer ]
    ppo_agent = PPO_CLIP(net_list, optimizers_list)
    print("# # # Agent Initialized # # #")
    p_info = np.zeros((1,59))
    np_info = np.zeros((1,145))
    print("# # # Agent policy testing # # #")
    action = ppo_agent.get_action([p_info, np_info]) 
    print(action)
    print("# # # Agent greedy policy testing # # #")
    result = ppo_agent.get_action_greedy([p_info, np_info]) 
    print(result)
    print("# # # Agent value function  testing # # #")
    result = ppo_agent.get_value([p_info, np_info]) 
    print(result)
    print("# # # Agent advantage calculation testing # # #")
    advantage = ppo_agent.cal_adv([p_info, np_info], 1.12)
    print("# # # Agent policy train testing # # #")
    cal_loss = ppo_agent.update([p_info, np_info], action, 0.25, 1, 1)

    print("# # # Test completed # # #")
    