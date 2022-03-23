import json
import numpy as np
from typing import *

# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)

# Obtenemos las constantes necesarias
ROBOT = ENV["ROBOT"]
H         : float = ROBOT["MAX_FOOT_HEIGHT"]
F_0       : float = ROBOT["GAIT_FREQUENCY"]
H_OFF     : float = ROBOT["H_OFF"]
V_OFF     : float = ROBOT["V_OFF"]
THIGH_L   : float = ROBOT["THIGH_LEN"]
SHANK_L   : float = ROBOT["SHANK_LEN"]
LEG_SPAN  : float = ROBOT["LEG_SPAN"]
H_OFF     : float = ROBOT["H_OFF"]
BASE_FREQ : float = ROBOT["BASE_FREQUENCY"]
SIGMA_0   : np.array = np.pi * np.asarray(ROBOT["INIT_FOOT_PHASES"])

SIM = ENV["SIMULATION"]
TERRAIN_FILE         : str         = SIM["TERRAIN_FILE"]
ROWS                 : int         = SIM["ROWS"]
COLS                 : int         = SIM["COLS"]
STEPS_FREQUENCY      : int         = SIM["STEPS_FREQUENCY"]
ZONE_STAIRS_WIDTH    : int         = SIM["ZONE_STAIRS_WIDTH"]
STEPS_NOISE          : float       = SIM["STEPS_NOISE"]
MESH_SCALE           : List[float] = SIM["MESH_SCALE"]
JOINTS_IDS           : List[int]   = SIM["JOINTS_IDS"]
HIPS_IDS             : List[int]   = SIM["HIPS_IDS"]
THIGHS_IDS           : List[int]   = SIM["THIGHS_IDS"]
SHANKS_IDS           : List[int]   = SIM["SHANKS_IDS"]
TOES_IDS             : List[int]   = SIM["TOES_IDS"] 
SIM_SECONDS_PER_STEP : float       = SIM["SIM_SECONDS_PER_STEP"]
EXTERNAL_FORCE_TIME  : float       = SIM["EXTERNAL_FORCE_TIME"]
EXTERNAL_FORCE_MAGN  : float       = SIM["EXTERNAL_FORCE_MAGN"]
SCALE                : float       = (MESH_SCALE[0] + MESH_SCALE[1]) / 2

PHYSICS = ENV["PHYSICS"]
VEL_TH         : float    = PHYSICS["VELOCITY_THRESHOLD"]
SWIGN_PH       : float    = PHYSICS["SWING_PHASE"]
H_Z            : np.array = np.asarray(PHYSICS["LEG_HORIZONTAL_Z_COMPONENT"])
GRAVITY_VECTOR : np.array = np.array(PHYSICS["GRAVITY_VECTOR"])

NN = ENV["NEURAL_NETWORKS"]
HISTORY_LEN                 : int = NN["HISTORY_LEN"]
FOOT_HISTORY_LEN            : int = NN["FOOT_HISTORY_LEN"]
JOINT_VEL_HISTORY_LEN       : int = NN["JOINT_VEL_HISTORY_LEN"]
JOINT_ERR_HISTORY_LEN       : int = NN["JOINT_ERR_HISTORY_LEN"]
PRIVILIGED_DATA_SHAPE       : int = NN["PRIVILIGED_DATA_SHAPE"]
NON_PRIVILIGED_DATA_SHAPE   : int = NN["NON_PRIVILIGED_DATA_SHAPE"]
CLASSIFIER_INPUT_DIFF_SHAPE : int = NN["CLASSIFIER_INPUT_DIFF_SHAPE"]
CLASSIFIER_INPUT_SHAPE      : int = CLASSIFIER_INPUT_DIFF_SHAPE + \
    NON_PRIVILIGED_DATA_SHAPE
NON_PRIVILIGED_DATA         : List[str] = NN["NON_PRIVILIGED_DATA"]
PRIVILIGED_DATA             : List[str] = NN["PRIVILIGED_DATA"]

ROS = ENV["ROS"]
QUEUE_SIZE : int = ROS["QUEUE_SIZE"]

TRAIN = ENV["TRAIN"]
MAX_ITER_TIME        : int   = TRAIN["MAX_ITERATION_TIME"]
N_TRAJ               : int   = TRAIN["N_TRAJ"]
N_EVALUATE           : int   = TRAIN["N_EVALUATE"]
N_PARTICLES          : int   = TRAIN["N_PARTICLES"]
GOAL_RADIUS_2        : float = TRAIN["GOAL_RADIUS"] ** 2
MIN_DESIRED_VEL      : float = TRAIN["MIN_DESIRED_VEL"]
P_REPLAY             : float = TRAIN["P_REPLAY"]
P_TRANSITION         : float = TRAIN["P_TRANSITION"]
MIN_DESIRED_TRAV     : float = TRAIN["MIN_DESIRED_TRAV"]
MAX_DESIRED_TRAV     : float = TRAIN["MAX_DESIRED_TRAV"]
RANDOM_STEP_PROP     : float = TRAIN["RANDOM_STEP_PROP"]


HILLS_RANGE      = ENV["HILLS_RANGE"]
STEPS_RANGE      = ENV["STEPS_RANGE"]
STAIRS_RANGE     = ENV["STAIRS_RANGE"]
