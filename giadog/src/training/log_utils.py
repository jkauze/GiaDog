"""
    Authors: Amin Arriaga, Eduardo Lopez
    Project: Graduation Thesis: GIAdog

    This file contains the code for the utilities (plots, logs, etc) for the 
    training of the robot.
    
    Reference:
    ----------
    https://stackoverflow.com/a/49414499


"""

import matplotlib.pyplot as plt
import numpy as np
# Make a function that 




def plot_reward_live(reward, first_exec = False):
    """
        Function to plot the cumulative reward, of each epoch during the 
        training.

        Arguments:
        ----------
        reward: list or np.array -> Cumulative reward of each epoch

        first_exec: bool -> If True, the plot will be created. 
                            If False, the plot will be updated.

        Returns:
        --------
        None
    """

    if first_exec:
        plt.ion()
        plt.xlabel('epoch')
        plt.ylabel('reward')

    plt.plot(reward)
    plt.xlabel('epoch')
    plt.ylabel('reward')
    plt.title('Cumulative reward')
    plt.draw()
    plt.pause(0.0001)
    plt.clf()


if __name__ == '__main__':
    """
    Testing the plot_reward_live function
    """
    reward = [0]

    plot_reward_live(reward, first_exec = True)
    for i in range(1000):
        reward.append(reward[-1] + np.random.random() * 0.90 **i)
        plot_reward_live(reward)
