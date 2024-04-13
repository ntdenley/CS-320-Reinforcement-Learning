# CS 320 Semester Project - Reinforcement Learning
**Contributed By:**
> Noah Denley, Igor Kovalenko, and Garett Bell

## Overview
This project began with the vision that we would be able to create and train a model for playing Minecraft, but this ended up being too ambitious with the time and experience that we had. Instead, we pivoted the project towards refining our view and learning about the backend, formulas, and parameters that all go into running and training a Reinforcement Learning based model. In this project, there are three separate "cool cams" or features, each independently developed by a member of the team. More info about these features in the **Cool Cams** section. The hopes in shifting our focus on the project was that we could have a more confident approach towards creating an actual Minecraft AI model if we had spent plenty of time creating interfaces and frameworks that utilize this technology on a simpler level.

## Cool Cams

**Noah Denley - Creating a DQN Model/Agent Interface w/ Plotting**
> The idea for my feature was relatively simple on paper: I wanted to create some kind of model and agent, then allow tweaking of parameters (such as learning rate, randomness, batch size, etc.) and then plotting the performance of these tweaks in order to visualize the differences in performance across multiple varying trials. Ultimately what this became was a foundation of a Deep Q Network build using pytorch, and an Agent that interfaces with and trains this model. The environment of choice was CartPole, a classic gymnasium training environment in which a cart is attempting to balance a pole for the maximum time. Once the DQN and Agent were both created, I created a tkinter window which would allow me to modify different parameters for the Agent and Network, then plot the results of their training as it happens, all while retaining plot information from previous runs.
> 
> From here, I would most likely try and get the interface to work in other gymnasium environments by making use of Garett's environment loader, then I would like to implement options for other RL based models such as AC2.

**Igor Kovalenko - Writing a Custom Tensor Library**
> Insert Igor's summary here!

**Garett Bell - Creating a GUI to Access and Load in Various Gymnasium Environments**
> Insert Garett's summary here!

## Tech Used
The project is written in Python 3, and mainly utilizes the following libraries:

- gymnasium (& box2d-py)
- matplotlib
- numpy
- torch

## How to Use:
Each feature is separated into their own directory, as we have not gotten to the stage of combining our features just yet. Noah's work is in `/dqn_plotting`, Igor's is in `/ml_framework`, and Garett's is in `/gui`. In order to ensure the right libraries are installed, use `pip install -r requirements.txt`, which references the requirements file in the root of the repository. From here, run the target file depending on your feature you wish to explore: 
> Noah - `./dqn_plotting/dqn_tweaks_gui.py` 