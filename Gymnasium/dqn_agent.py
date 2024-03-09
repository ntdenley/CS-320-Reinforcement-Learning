'''
This file contains a Deep Q learning based algorithm for agents to be
trained on.
'''

import torch
import torch.nn as nn
import torch.optim as optim

import random
import math
from collections import namedtuple, deque
from dqn_model import DQN

# Create a named tuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128 # How many transitions to sample from memory
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # Beginning epsilon value - chance for random action to occur
EPS_END = 0.05 # Ending epsilon value
EPS_DECAY = 1000 # How quickly the epsilon value will decay
TAU = 0.005 # Update rate of target network
LR = 1e-4 # Learning rate for the optimizer

"""
The Agent:
    The Agent is responsible for interfacing with the neural network model and the environment,
    getting actions through envrionment state, improving the network, and keeping track of
    previous states, rewards, and actions to be used for further optimization. Most of the actual
    algorithm applies here, since the Model itself is simply the template, while the Agent forms
    it through optimization.
"""
class DQN_Agent():

    def __init__(self, env):

        self.env = env
        # Get number of actions from gym action space
        n_actions = env.action_space.n
        # Get the number of state observations
        state, info = env.reset()
        n_observations = len(state)

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.steps_done = 0

    # This function simply pushes environment information into the memory
    # queue of the agent.
    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    # This function updates the target network using the state dictionary
    # of the policy network, based on the update value TAU. Having a policy
    # network and a target network allows more stability in the learning,
    # since updates to the target network are gradually applied.
    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def optimize_model(self):

        # Only perform if the memory is big enough to fill the batch.
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Gather transitions by sampling a batch from the memory
        transitions = self.memory.sample(BATCH_SIZE)

        # The resulting transitions are a batch-array, which needs to be unzipped.
        # this is also referred to as transposing, which is explained further here:
        #  ->   https://stackoverflow.com/a/19343/3343043
        batch = Transition(*zip(*transitions))

        # Create a boolean mask which shows which states in the batch are final states
        # (game over) or non-final states (game still in progress).
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        # Create tensors from the batch of next states that are non-final
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Create tensors which combine all of the states, actions, and rewards in the batch into
        # their own combined tensor
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        # 
        # NOTE: by inputting the state batch into the policy network, all of the individual states
        #       are parallelized and applied through the network by pytorch automatically.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Gradients are clipped to prevent the gradients from becoming too large.
        # 100 is the max gradient magnitude assigned here.
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def select_action(self, state):
        # get random number between [0,1)
        sample = random.random()
        # decay epsilon based on number of steps completed
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)

        # increase step counter
        self.steps_done += 1
        if sample > eps_threshold:
            # torch.no_grad is a context manager which disables gradient computation within
            # the contained block. This is necessary because we are getting an action from
            # the model, but do not wish to modify it in any way.
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # Random action - "Explore"
            return torch.tensor([[self.env.action_space.sample()]], device=device, dtype=torch.long)


# Replay Memory, a buffer that will hold transitions recently observed, which contains a
# sample() method for selecting a random batch of transitions (used for training).
class ReplayMemory(object):

    def __init__(self, capacity):
        # Initialize a deque, which is similar to a stack, but double-ended.
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        # Push a transition tuple into the memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Use random.sample(S, n), which gives n samples from sequence S
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Returns the size of the deque.
        return len(self.memory)