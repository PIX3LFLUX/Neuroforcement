import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from Modules import Visualization as Vis

###################################################################################################
class FrozenLake:
    """Class for use with the Gym Frozen Lake Environment"""
    """Possible Versions: FrozenLake-v0 or FrozenLake8x8-v0 """
    def __init__(self, version):
        self.env = gym.make(version, is_slippery = False)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
    
###################################################################################################  
class Taxi:
    """Class for use with the Gym Frozen Lake Environment"""
    def __init__(self, version):
        self.env = gym.make(version)
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n

###################################################################################################
class CartPole:
    """Class for use with the Cart Pole Environment"""
    def __init__(self, version, display = None, image_observation = False):
        if image_observation:
            self.env = gym.make(version).unwrapped
            self.display = display
            self.resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
            # Get screen size so that we can initialize layers correctly based on shape
            # returned from AI gym. Typical dimensions at this point are close to 3x40x90
            # which is the result of a clamped and down-scaled render buffer in get_screen()
            self.env.reset()
            self.init_screen = self.get_screen()
            _, _, self.screen_height, self.screen_width = self.init_screen.shape
        else:
            self.env = gym.make(version)
            #self.observation_space = self.env.observation_space.shape[0]
            #self.n_actions = self.env.action_space.n
        
    def get_screen(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        # such as 800x1200x3. Transpose it into torch order (CHW).
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.4)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        # Resize, and add a batch dimension (BCHW)
        resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])
    
        return resize(screen).unsqueeze(0)
    
    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

###################################################################################################
class MountainCar:
    """Class for use with the Mountain Car Environment"""
    def __init__(self, version):
        self.env = gym.make(version)
        self.observation_space = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n