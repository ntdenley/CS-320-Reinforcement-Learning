'''
This code is meant to demonstrate how plots will be shown and updated as training is completed by the agent.

The learning code and methods are followed from a blog about Reinforcement Q-Learning:
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

With some adaptations for the newer version of gym and to allow for rendering post-training.
'''

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import random

# Initialize the environment with human render mode so it can be visible.
env = gym.make("Taxi-v3")

# Initialize empty q table.
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Enable interactive mode for the plot
plt.ion()

# Hyperparameters
alpha = 0.1 # Training "strength" value
gamma = 0.6 # Secondary training "strength" value
epsilon = 0.1 # Take random action 10% of the time.

# Plot variables
plot_step = 10 # Adjusts how often data is plotted - lower number = slower but more precise data.
penalty_list = []
xs = []
plot = None

# Training Loop
for i in range(1, 2500):
    # Get initial info
    state, info = env.reset()

    # Set env variables
    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    # Game loop
    while not done:
        # Random chance of taking random action
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        # Get info from environment
        next_state, reward, done, truncated, info = env.step(action) 
        
        # Adjust q value based on current state
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])   
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        # Dectect penalties
        if reward == -10:
            penalties += 1

        state = next_state
        
    if i % 100 == 0:
        print(f"Episode: {i}")

    # Plotting
    if i % plot_step == 0:

        # Add penalty count
        penalty_list.append(penalties)
        xs = [x for x in range(0, len(penalty_list)*plot_step, plot_step)]

        # Update Plot
        if plot is not None:
            plot.remove()
        plot = plt.plot(xs, penalty_list, color = 'r')[0]
        plt.xlim(xs[0], xs[-1])
        plt.pause(0.0001)

print("Training finished.\n")

# Remake environment for human eyes
env = gym.make("Taxi-v3", render_mode="human")

episodes = 100

# Post-Training loop
for _ in range(episodes):
    state, info = env.reset()
    
    done = False
    
    while not done:

        # Get action from q-table
        action = np.argmax(q_table[state])
        state, reward, done, truncated, info = env.step(action)
