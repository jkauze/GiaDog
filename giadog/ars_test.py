

from src.training.GiadogGym import *
from src.training.ARS import ARS
sim = Simulation('giadog/mini_ros/urdf/spot.urdf', gui=True)
gym_env = TeacherEnv(sim)

state_features = [  'command_dir', 'turn_dir', 
                    'gravity_vector', 'angular_vel', 
                    'linear_vel', 'joint_angles', 
                    'joint_vels', 'ftg_phases', 
                    'ftg_freqs', 'base_freq', 
                    'joint_err_hist', 'joint_vel_hist', 
                    'feet_target_hist', 'toes_contact']


agent = ARS(
                gym_env,# gym enviroment
                state_features,# state features
                0.02, # step size
                1, # sample direction per iteration
                0.03, # exploration std deviation noise
                1, # number of top direction to use
                2000,# number of exploration steps
            )


print(agent.normalizer.dmu)
agent.save_model("models/ars_test.npz")
agent.normalizer.dmu = 12
print(agent.normalizer.dmu)
agent.load_model("models/ars_test.npz")
print(agent.normalizer.dmu)
for i in range(1):
    agent.update_V2("terrains/initial_hills.txt")

