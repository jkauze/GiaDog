import os, sys
sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))

import pathlib
from unicodedata import name
from uuid import uuid4
from src.__env__ import TERRAIN_FILE
from src.simulation.Simulation import *
from src.simulation.terrain_gen import *

import time

if __name__ == '__main__':
    ROWS = COLS = 500

    rough_id = 0
    freq_id = 0.2
    amp_id =0.2
    file = f'hills_{uuid4()}.txt'

    terrain = hills(
                ROWS,
                COLS,
                rough_id,
                freq_id,
                amp_id,
                randint(0, 1e6)
            )

    spot_urdf = str(pathlib.Path(__file__).parent.parent.resolve()) + \
                                    '/mini_ros/urdf/spot.urdf'
    sim = Simulation(spot_urdf, gui=True)
    save_terrain(terrain, file)
    sim.reset(file)

    r_o = sim.position
    roll, pitch, yaw = sim.orientation

    x = np.cos(yaw)*np.cos(pitch)
    y = np.sin(yaw)*np.cos(pitch)
    z = np.sin(pitch)
    r_f = r_o + np.array([x, y, z])
    vector = sim.draw_vector(r_o, r_f)
    dt = 1/240
    while True:
        sim.update_position_orientation()
        orientation = sim.p.getQuaternionFromEuler(sim.orientation)
        sim.p.resetBasePositionAndOrientation(vector,sim.position,orientation)
        sim.step()
        time.sleep(dt)