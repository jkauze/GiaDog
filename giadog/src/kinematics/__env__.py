import json
import numpy as np
from typing import *

# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)

H        : float = ENV["ROBOT"]["MAX_FOOT_HEIGHT"]
F_0      : float = ENV["ROBOT"]["GAIT_FREQUENCY"]
H_OFF    : float = ENV["ROBOT"]["H_OFF"]
V_OFF    : float = ENV["ROBOT"]["V_OFF"]
THIGH_L  : float = ENV["ROBOT"]["THIGH_LEN"]
SHANK_L  : float = ENV["ROBOT"]["SHANK_LEN"]
LEG_SPAN : float = ENV["ROBOT"]["LEG_SPAN"]
H_OFF    : float = ENV["ROBOT"]["H_OFF"]
H_Z      : np.array = np.asarray(ENV["PHYSICS"]["LEG_HORIZONTAL_Z_COMPONENT"])
SIGMA_0  : np.array = np.pi * np.asarray(ENV["ROBOT"]["INIT_FOOT_PHASES"])