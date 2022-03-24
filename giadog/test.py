from re import X
from src.simulation import *
from src.training import *
from src.agents import *
from src.PPO import PPO_CLIP
import pybullet as p
import numpy as np
from tensorflow.keras.optimizers import Adam
from src.__env__ import NON_PRIVILIGED_DATA_SHAPE, PRIVILIGED_DATA_SHAPE

if __name__ == '__main__':
    # Create simulation
    spot_urdf_file = 'giadog/mini_ros/urdf/spot.urdf'
    sim = Simulation(spot_urdf_file, p,
                self_collision_enabled=False)

    
    env = TeacherEnv(sim = sim)

    print("# # # Env Initialized # # #")
    print("priviliged_data_shape: ", PRIVILIGED_DATA_SHAPE)
    print("non_priviliged_data_shape: ", NON_PRIVILIGED_DATA_SHAPE)
    print("# # # # # # # # # # # # #")
    ac_space = env.action_space
    obs_space = env.observation_space
    policy = TeacherNetwork(action_space = ac_space,
                        observation_space = obs_space)
    value_function = TeacherValueNetwork(obs_space)
    print("# # # Policy Initialized # # #")
    policy_optimizer = Adam(lr=0.001)
    value_function_optimizer = Adam(lr=0.001)
    print("# # # Optimizers Initialized # # #")
    net_list = [value_function, policy]
    optimizers_list = [value_function_optimizer ,policy_optimizer ]

    print("### Testing PPO_CLIP Agent ####")
    ppo_agent = PPO_CLIP(net_list, optimizers_list)
    print("# # # Agent Initialized # # #")
    states = [obs_space.sample() for _ in range(5)]
    print("# # # Agent policy testing # # #")
    action = ppo_agent.get_action(states) 
    print(action)
    print("# # # Agent greedy policy testing # # #")
    result = ppo_agent.get_action_greedy(states) 
    print(result)
    print("# # # Agent value function  testing # # #")
    result = ppo_agent.get_value(states) 
    print(result)
    print("# # # Agent advantage calculation testing # # #")
    advantage = ppo_agent.cal_adv(states, np.random.randn(5, 1))
    #print(advantage)
    print("# # # Agent policy train testing # # #")
    cal_loss = ppo_agent.update(states, np.random.randn(5, 16), np.random.randn(5, 1), 5, 5)
    print("# # # Test completed # # #")
    
    print("### Testing TRPO Agent ####")
    trpo_agent = TRPO(
        TeacherNetwork(env.action_space, env.observation_space), 
        TeacherValueNetwork(env.observation_space),
        Adam(lr=0.001)
    )
    print("# # # Agent Initialized # # #")
    states = [obs_space.sample() for _ in range(5)]
    print("# # # Agent policy testing # # #")
    action = trpo_agent.get_action(states[0]) 
    print(action)
    print("# # # Agent greedy policy testing # # #")
    result = trpo_agent.get_action_greedy(states[0]) 
    print(result)
    print("# # # Agent value function  testing # # #")
    result = trpo_agent.critic_value(states[0]) 
    print(result)
    print("# # # Agent advantage calculation testing # # #")
    advantage = trpo_agent.advantage(states, np.random.randn(5, 1))
    print(advantage)
    print("# # # Agent policy train testing # # #")
    cal_loss = trpo_agent.update(states, np.random.randn(5, 16), np.random.randn(5, 1), 5, 5, 0.24)
    print("# # # Test completed # # #")
    