import json
import numpy as np
from typing import *

# Cargamos las variables de entorno
with open('.env.json', 'r') as f:
    ENV = json.load(f)

# Obtenemos las constantes necesarias
STEPS_FREQUENCY   : int         = ENV["SIMULATION"]["STEPS_FREQUENCY"]
ZONE_STAIRS_WIDTH : int         = ENV["SIMULATION"]["ZONE_STAIRS_WIDTH"]
STEPS_NOISE       : float       = ENV["SIMULATION"]["STEPS_NOISE"]
MESH_SCALE        : List[float] = ENV["SIMULATION"]["MESH_SCALE"]
SCALE             : float       = (MESH_SCALE[0] + MESH_SCALE[1]) / 2
LEG_SPAN          : float       = ENV["ROBOT"]["LEG_SPAN"]
H_OFF             : float       = ENV["ROBOT"]["H_OFF"] 
GRAVITY_VECTOR    : np.array    = np.array(ENV["PHYSICS"]["GRAVITY_VECTOR"])
JOINTS_IDS        : List[int]   = ENV["SIMULATION"]["JOINTS_IDS"]
HIPS_IDS          : List[int]   = ENV["SIMULATION"]["HIPS_IDS"]
THIGHS_IDS        : List[int]   = ENV["SIMULATION"]["THIGHS_IDS"]
SHANKS_IDS        : List[int]   = ENV["SIMULATION"]["SHANKS_IDS"]
TOES_IDS          : List[int]   = ENV["SIMULATION"]["TOES_IDS"] 
SIM_SECONDS_PER_STEP : float = ENV["SIMULATION"]["SIM_SECONDS_PER_STEP"]
EXTERNAL_FORCE_TIME  : float = ENV["SIMULATION"]["EXTERNAL_FORCE_TIME"]
EXTERNAL_FORCE_MAGN  : float = ENV["SIMULATION"]["EXTERNAL_FORCE_MAGN"]