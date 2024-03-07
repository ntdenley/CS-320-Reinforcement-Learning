'''
This python script will be the basis for setting up user input for
interfacing with the plotting and training. In the future, it will
be based on GUI input rather than CLI input.
'''
# Import environments
import importlib

# Get choice env from the user
choice = ""
choices = ["taxi_qlearning"]
while choice not in choices:
    choice = input("Please choose an environment to load (type \"help\" to list environments): ")
    if choice == "help":
        print(*choices, sep=", ")

# Get config options
episodes = int(input("How many episodes should the AI run?: "))
average_enabled = input("Display average plot? (y/n | default 'n'): ") == 'y'

# import and run proper env file
env = importlib.import_module(choice)

env.run_env(episodes, average_enabled)