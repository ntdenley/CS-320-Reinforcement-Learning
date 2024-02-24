'''
This file contains a Deep Q learning based algorithm for agents to be
trained on.
'''

import torch
import random
import numpy as np
from collections import deque
from dqn_model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class DQN_Agent():

    def __init__(self, num_games, env):
        self.num_games = num_games
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (between 0-1)
        self.memory = deque(maxlen=MAX_MEMORY)

        self.env = env
        self.obs_size = self.env.observation_space.n
        
        # Initialize the NN based on the observation space and action space size,
        # with an arbitrary hidden layer size (256 here.)
        self.model = Linear_QNet(env.observation_space.n, 256, env.action_space.n)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def train_long(self):
        pass 

    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Use epsilon to determine if a random action should be taken.
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            encoded_obs = np.zeros(self.obs_size)
            encoded_obs[state] = 1
            
            # Converts the current state into a tensor with float dtype
            state_tensor = torch.tensor(encoded_obs, dtype=torch.float)

            # self.model() will call the model.forward method, which gives us the output layer's result.
            model_prediction = self.model(state_tensor)

            # The max value node in the output layer is our action of choice.
            action = torch.argmax(model_prediction).item()
            
        return action