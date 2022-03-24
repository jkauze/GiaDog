from re import X
from src.simulation import *
from src.training import *
from src.agents import *
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

    print("### Testing PPO_CLIP Agent ####")
    ppo_agent = PPO(
        TeacherNetwork(env.action_space, env.observation_space), 
        TeacherValueNetwork(env.observation_space),
        Adam(lr=0.001),
        Adam(lr=0.001)
    )
    print("# # # Agent Initialized # # #")
    states = [obs_space.sample() for _ in range(5)]
    print("# # # Agent policy testing # # #")
    action = ppo_agent.get_action(states[0]) 
    print(action)
    print("# # # Agent greedy policy testing # # #")
    result = ppo_agent.get_action_greedy(states[0]) 
    print(result)
    print("# # # Agent value function  testing # # #")
    result = ppo_agent.critic_value(states[0]) 
    print(result)
    print("# # # Agent advantage calculation testing # # #")
    advantage = ppo_agent.advantage(states, np.random.randn(5, 1))
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
    