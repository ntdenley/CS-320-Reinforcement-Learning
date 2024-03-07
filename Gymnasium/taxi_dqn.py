import gymnasium as gym
from dqn_agent import DQN_Agent
import numpy as np
import random
from gymplots import aiPlot

def begin_training(episodes, calc_avg):
    env = gym.make("Taxi-v3")

    agent = DQN_Agent(10000, env)

    graph = aiPlot(step_value=1000, calculate_avg=calc_avg)

    env.reset()

    # Training loop
    for i in range(1, episodes):
        state, info = env.reset()

        penalties = reward = 0
        done = truncated = False


        while not (done or truncated):
            action = agent.get_action(state)

            next_state, reward, done, truncated, info = env.step(action)

            agent.train_short(state, action, reward, next_state, done)

            agent.remember(state, action, reward, next_state, done)

            if reward == -10:
                penalties += 1
            
            state = next_state

        #agent.train_long()
        graph.update(i, penalties)

    print("Training finished.\n")

    # Remake environment for human eyes
    env = gym.make("Taxi-v3", render_mode="human")

    # Post-Training loop
    for _ in range(100):
        state, info = env.reset()
        done = False
        
        while not done:

            # Get action from q-table (determined by max col value in the row)
            action = agent.get_action(state)
            state, reward, done, truncated, info = env.step(action)


if __name__ == "__main__":
    begin_training(10000, True)