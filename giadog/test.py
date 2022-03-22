from re import X
from src.simulation import *
from src.giadog_gym import teacher_giadog_env 
from src.agents import *
from src.PPO import PPO_CLIP
from src.TRPO import TRPO
import pybullet as p
import numpy as np
from tensorflow.keras.optimizers import Adam


if __name__ == '__main__':
    # Create simulation
    spot_urdf_file = 'giadog/mini_ros/urdf/spot.urdf'
    sim = Simulation(spot_urdf_file, p,
                self_collision_enabled=False)

    
    env = teacher_giadog_env(sim = sim)
    print("# # # Env Initialized # # #")
    print("priviliged_data_shape: ", env.privileged_space_shape)
    print("non_priviliged_data_shape: ", env.non_privileged_space_shape)
    print("# # # # # # # # # # # # #")
    ac_space = env.action_space
    obs_space = env.observation_space
    policy = teacher_network(action_space = ac_space,
                        observation_space = obs_space)
    value_function = TeacherValueNetwork()
    print("# # # Policy Initialized # # #")
    policy_optimizer = Adam(lr=0.001)
    value_function_optimizer = Adam(lr=0.001)
    print("# # # Optimizers Initialized # # #")
    net_list = [value_function, policy]
    optimizers_list = [value_function_optimizer ,policy_optimizer ]

    print("### Testing PPO_CLIP Agent ####")
    ppo_agent = PPO_CLIP(net_list, optimizers_list)
    print("# # # Agent Initialized # # #")
    p_info = np.random.randn(5, 59)
    np_info = np.random.randn(5, 145)
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
    advantage = ppo_agent.cal_adv([[p_info, np_info]], np.random.randn(5, 1))
    #print(advantage)
    print("# # # Agent policy train testing # # #")
    cal_loss = ppo_agent.update([p_info, np_info], np.random.randn(5, 16), np.random.randn(5, 1), 5, 5)
    print("# # # Test completed # # #")
    
    print("### Testing TRPO Agent ####")
    net_list = [value_function, policy]
    optimizers_list = [value_function_optimizer]# ,policy_optimizer ]
    trpo_agent = TRPO(net_list, optimizers_list)
    print("# # # Agent Initialized # # #")
    p_info = np.random.randn(5, 59)
    np_info = np.random.randn(5, 145)
    print("# # # Agent policy testing # # #")
    action = trpo_agent.get_action([p_info, np_info]) 
    print(action)
    print("# # # Agent greedy policy testing # # #")
    result = trpo_agent.get_action_greedy([p_info, np_info]) 
    print(result)
    print("# # # Agent value function  testing # # #")
    result = trpo_agent.get_value([p_info, np_info]) 
    print(result)
    print("# # # Agent advantage calculation testing # # #")
    advantage = trpo_agent.cal_adv([[p_info, np_info]], np.random.randn(5, 1))
    #print(advantage)
    print("# # # Agent policy train testing # # #")
    cal_loss = trpo_agent.update([p_info, np_info], np.random.randn(5, 16), np.random.randn(5, 1), 5, 5, 0.24)
    print("# # # Test completed # # #")
    