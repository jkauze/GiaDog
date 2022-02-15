#!/usr/bin/env python3
import rospy
import numpy as np
from time import sleep
from random import randint
from src.giadog_gym import *

if __name__ == '__main__':
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
    sleep(2)

    while True:
        done = False
        obs = train_env.get_obs()
        while not done:
            # Obtenemos la accion de la politica
            action = train_env.predict(obs)
            # Aplicamos la accion al entorno
            obs, reward, done, info = train_env.step(action)

        tr = train_env.traversability()
        print(f'Traversatility: {tr}')

        # Reseteamos el terreno
        train_env.reset('gym_terrain.txt')
        sleep(2)

