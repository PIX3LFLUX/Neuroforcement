import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import pynvml as py 

from collections import namedtuple, deque

###################################################################################################
# Helper function to find the GPU with max. free memory left
def get_gpu_with_max_free_mem():
    py.nvmlInit()
    mem_free = np.zeros(torch.cuda.device_count())
    for gpu_index in range(torch.cuda.device_count()):
        handle = py.nvmlDeviceGetHandleByIndex(int(gpu_index))
        mem_info = py.nvmlDeviceGetMemoryInfo(handle)
        mem_free[gpu_index] = mem_info.free // 1024 ** 2
        
    GPU_num = np.argmax(mem_free)
    return GPU_num

def flatten(x):
    N = x.shape[0] 
    return x.view(N, -1)

###################################################################################################
class QlearningAgent:
    """All the necessary functions to get a simple Q-learning Agent that stores the Q-function in a Table"""
    def __init__(self, n_states, n_actions, epsilon, lr_rate, gamma):
        self.epsilon = epsilon
        self.lr_rate = lr_rate
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
    
    def choose_action(self, state):
        action=0
        if np.random.uniform(0, 1) < self.epsilon:
            return "random"
        else:
            action = np.argmax(self.Q[state, :])
            return action
    
    def learn(self, state, state2, reward, action):
        predict = self.Q[state, action]
        target = reward + self.gamma * np.max(self.Q[state2, :])
        self.Q[state, action] = self.Q[state, action] + self.lr_rate * (target - predict)
        
###################################################################################################   
class DQNAgent_observations:
    """All the necessary functions to get a Deep-Q-learning Agent which uses the Observations given from Open Ai gym as inputs and estimates the Q-Function with a Neural Network"""
    def __init__(self, exploration_max, exploration_decay, exploration_min, environment, mem_size, batch_size, gamma):
        
        # run on the GPU with max free ram, if cuda is available
        self.device = torch.device(('cuda:' + str(get_gpu_with_max_free_mem())) if torch.cuda.is_available() else "cpu")

        self.env = environment # remember the environment
        self.memory = ReplayBuffer(self.env, mem_size, batch_size) # init the replay buffer     
        
        # remember the hyperparameters
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        self.exploration_rate = exploration_max
        self.gamma = gamma
        
        FC1_DIMS = 1024
        FC2_DIMS = 512
        self.network = Network(self.env, self.device, FC1_DIMS, FC2_DIMS) # initialize the network
        print('using device:', self.device)

    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample() # perform random action        
        state = torch.tensor(observation).float().detach() # simply convert to a tensor
        state = state.to(self.device) # move data to device, e.g. GPU
        state = state.unsqueeze(0)
        q_values = self.network(state) # calculate the q-values
        return torch.argmax(q_values).item() # select the action based on the highes q-value
    
    def learn(self):
        if self.memory.mem_count < self.batch_size: # only learn after #batchsize data is available in memory
            return
        
        # read data for the training in size batchsize from memory
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        batch_indices = np.arange(self.batch_size, dtype=np.int64) # number the sampled data samples of the batch

        # get the q-values for all the states and nextstates for the data from memory
        q_values = self.network(states)
        next_q_values = self.network(states_)
        
        predicted_value_of_now = q_values[batch_indices, actions] # select the q-values of the states based on the action
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0] # select the future q-values based on the max. values
        
        # calculate the target q-value based on the future predicted values and the reward
        q_target = rewards + self.gamma * predicted_value_of_future * dones 

        loss = self.network.loss(q_target, predicted_value_of_now) # calculate the MSE between the target and predicted values
        self.network.optimizer.zero_grad() # reset the networks gradient
        loss.backward() # calculate the network gradient
        self.network.optimizer.step() # optimize one step

        self.exploration_rate *= self.exploration_decay # decay the exploration value
        self.exploration_rate = max(self.exploration_min, self.exploration_rate) # ensure the the exploration value stays above min exploration

    def returning_epsilon(self):
        return self.exploration_rate
    
# We need to wrap `flatten` function in a module in order to stack it in nn.Sequential
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
    
class Network(torch.nn.Module):
    def __init__(self, env, device, layer1_out, layer2_out):
        super().__init__()
        # the network is a simple fully connected network with two hidden layers. 
        # the input has the same size then the observation space
        # the output has the same size then the action space
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n
        self.fc1 = nn.Linear(*self.input_shape, layer1_out)
        self.fc2 = nn.Linear(layer1_out, layer2_out)
        self.fc3 = nn.Linear(layer2_out, self.action_space)

        self.optimizer = optim.AdamW(self.parameters(), lr=0.001) # using adamw as optimizer
        self.loss = nn.MSELoss() # MSEloss
        self.to(device) # store network on the device e.g. GPU
    
    def forward(self, x):
        # this defines the data flow through the network, 
        # the data passes from the input through the two layers to the output.
        # Thereby the observation is is transformed to the q-values
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer():
    def __init__(self, env, mem_size, batch_size):
        self.mem_count = 0 # how many items are stored
        self.mem_size = mem_size # store the mem-size
        self.batch_size = batch_size # store the batchsize
        
        self.states = np.zeros((self.mem_size, *env.observation_space.shape),dtype=np.float32) # init the state storage
        self.actions = np.zeros(self.mem_size, dtype=np.int64) # init the action storage
        self.rewards = np.zeros(self.mem_size, dtype=np.float32) # init the reward storage
        self.states_ = np.zeros((self.mem_size, *env.observation_space.shape),dtype=np.float32) # init the next-state storage
        self.dones = np.zeros(self.mem_size, dtype=bool) # init the storage where the agent reaches the terminal state 
    
    def add(self, state, action, reward, state_, done):
        # simply store the returns of a step 
        mem_index = self.mem_count % self.mem_size # implements a ringbuffer of size "self.mem_size"
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        
        MEM_MAX = min(self.mem_count, self.mem_size) # makes sure that only returns from steps out of stored memory is read
        
        batch_indices = np.random.choice(MEM_MAX, self.batch_size, replace=True) #CHANGE 64 TO BATCH SIZE
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones
    
###################################################################################################   
class DQNAgent_image:
    """All the necessary functions to get a Deep-Q-learning Agent which uses the Pixel Data as inputs and estimates the Q-Function with a Neural Network"""
    def __init__(self, exploration_max, exploration_decay, exploration_min, env, batch_size, gamma):
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = exploration_max
        self.EPS_END = exploration_min
        self.EPS_DECAY = exploration_decay
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = env.env.action_space.n
        
        self.policy_net = Image_Network(env.screen_height, env.screen_width, self.n_actions, self.device).to(self.device)
        self.target_net = Image_Network(env.screen_height, env.screen_width, self.n_actions, self.device).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=0.001) # using adamw as optimizer
        self.memory = ReplayMemory(10000)
        
    def select_action(self, state, steps_done):
        sample = random.random()
        self.EPS_START *= self.EPS_DECAY
        self.EPS_START = max(self.EPS_START, self.EPS_END)
        if sample > self.EPS_START:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
    
        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Image_Network(nn.Module):
    def __init__(self, h, w, outputs, device):
        super(Image_Network, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 128
        half_linear_input_size = int(linear_input_size/2)
        self.fc1 = nn.Linear(linear_input_size, half_linear_input_size)
        self.head = nn.Linear(half_linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization.
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)
###################################################################################################
