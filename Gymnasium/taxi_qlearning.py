'''
This code is meant to demonstrate how plots will be shown and updated as training is completed by the agent.

The learning code and methods are followed from a blog about Reinforcement Q-Learning:
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

With some adaptations for the newer version of gym and to allow for rendering post-training.
'''
import gymnasium as gym
import numpy as np
import random
from gymplots import aiPlot

def run_env(episodes, calc_avg):
    # Initialize the environment with human render mode so it can be visible.
    env = gym.make("Taxi-v3")

    # Initialize empty q table.
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    # Hyperparameters
    alpha = 0.1 # Learning rate
    gamma = 0.6 # Discount factor - determines value of previous rewards (60%)
    epsilon = 0.1 # Take random action 10% of the time.

    # Initialize plot which updates every 1000 frames and calculates averages
    graph = aiPlot(step_value=1000, calculate_avg=calc_avg)

    # Training Loop
    for i in range(1, episodes):
        # Get initial info
        state, info = env.reset()

        # Set env variables
        epochs, penalties, reward = 0, 0, 0
        done, truncated = False, False
        
        # Game loop
        while not (done or truncated):
            # Random chance of taking random action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            # Get info from environment
            next_state, reward, done, truncated, info = env.step(action) 
            
            # Adjust q value based on current state and previous state
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])   
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            # Dectect drop off/pick up penalties
            if reward == -10:
                penalties += 1

            state = next_state

        graph.update(i, penalties) # Update graph with new values

    print("Training finished.\n")

    # Remake environment for human eyes
    env = gym.make("Taxi-v3", render_mode="human")

    # Post-Training loop
    for _ in range(100):
        state, info = env.reset()
        done = False
        
        while not done:

            # Get action from q-table (determined by max col value in the row)
            action = np.argmax(q_table[state])
            state, reward, done, truncated, info = env.step(action)
