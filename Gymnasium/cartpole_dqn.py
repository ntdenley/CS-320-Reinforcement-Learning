"""
Re-implementation of the DQN attempted previously. This one follows closely with a tutorial
posted on pytorch.org, link below. The code has been modified to work with my plotting interface.
Since this implementation is much closer to the actual source code of the tutorial, I've made an effort
to comment and break down this code with comments as much as possible so that I can understand what this
code does on a line-by-line basis.

link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import gymnasium as gym
from itertools import count
from dqn_agent import DQN_Agent, device
from gymplots import aiPlot
import torch

def run(params):
    env = gym.make("CartPole-v1")

    agent = DQN_Agent(env, params)
    graph = aiPlot(step_value=1, calculate_avg=True)

    num_episodes = params["episode count"]

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            agent.soft_update()

            if done:
                graph.update(i_episode, t)
                break

    env = gym.make("CartPole-v1", render_mode="human")
    for ii_episode in range(params["episode display count"]):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            agent.remember(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # # Perform one step of the optimization (on the policy network)
            # agent.optimize_model()

            # # Soft update of the target network's weights
            # # θ′ ← τ θ + (1 −τ )θ′
            # agent.soft_update()

            if done:
                graph.update(i_episode, t)
                break
    env.close()
    graph.close()
