import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
import numpy as np
import torch
import time

"""Functions to start the virtual display and to render the rgb image from gym"""
    
def start_display():
    # Starting the Virtual Display to allow environments to be rendered and return it to the main
    # Call this function once at the beginning of the script
    display = Display(visible=0, size=(1400, 900))
    display.start()
        
    is_ipython = 'inline' in plt.get_backend()
    if is_ipython:
        from IPython import display     
    plt.ion()
    return display
        
def output_frame(render, display, plot, episode = None, step_counter = None):
    """
    Plot the Output of the Gym Environment
    Call this Function for each action-/timestep to output the changing environment
    
    Arguments:
    render = output of the render(mode='rgb_array') function of open ai gym
    display = a pyvirtual display as returned by the start_display() function
    plot = matplotlib image plot surface returned by imshow() function of matplotlib
    episode = integer number for which episode is currently displayed
    step_counter = which action-/timestep is currently beeing displayed
    """
    plot.set_data(render)
    if episode is not None and step_counter is not None:
        plt.title("Episode: %d, Step: %d" % (episode, step_counter))
    plt.axis('off')
    display.display(plt.gcf())
    display.clear_output(wait=True)
    

    
###################################################################################################
class Plotting:
    def __init__(self):
        self.Episode_number = []
        self.Episode_reward = []
        self.Average_reward = [] 
           
    def plot_reward(self):
        plt.close()
        fig2 = plt.figure()
        plt.clf()
        plt.plot(self.Episode_number, self.Episode_reward, label='Episode Reward')
        plt.plot(self.Episode_number, self.Average_reward, label='Average Reward')
        plt.title('Reward and Average Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend(loc=2)
        plt.show()
        
    def add_data(self, episode_number, episode_reward):
        
        self.Episode_number.append(episode_number)
        self.Episode_reward.append(episode_reward)
        self.Average_reward.append(np.sum(self.Episode_reward)/episode_number)