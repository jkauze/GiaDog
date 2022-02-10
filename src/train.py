#!/usr/bin/env python3
import rospy
from src.giadog_gym import *

if __name__ == '__main__':
    import numpy as np
    from time import sleep
    from random import randint


    rospy.init_node('train', anonymous=True)
    train_env = teacher_giadog_env()
    train_env.make_terrain(
        'hills',
        rows=500,
        cols=500,
        roughness=0.0,
        frequency=0.2,
        amplitude=0.2,
        seed=randint(0, 1e6)
    )
    train_env.reset('gym_terrain.txt')

    sleep(5)
    while True:
        print(train_env.step(np.array([0]*16))[0])
        sleep(1)

