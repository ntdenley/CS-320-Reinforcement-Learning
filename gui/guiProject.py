'''
    Imports
        - Gym (Gymnasium): OpenAI Environment Library
        - Pygame: Multi Media Game Creation Library
        - Numpy: Numerical Operation Library
        - tkinter (tk, ttk): GUI Library
        - Pillow: Rendering Library
        - os: Operating System Library
        - threading: Thread Library for running multiple processes
'''
import tkinter as tk
import pygame
import gym
import os
import threading
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
from gymplots import aiPlot 

'''
    GUI Class:
        Initalizes and handles aspects of the main Tkinter GUI window and the OpenAI environments.
'''
class EnvironmentGUI:
    '''
        Init Function: constructor method for GUI environment and initalizes instance of the class with a Tkinter root window.
            Parameters:
                - self: instance of class object
                - root: Tkinter root window
    '''
    def __init__(self, root):
        self.root = root
        self.setupGUI()

    '''
        setupGUI Function: Used to set up the graphical user interface (GUI) by providing the dimensions/title for the Tkinter root window. 
            Parameters:
                - self: instance of class object
    '''
    def setupGUI(self):
        # Set title and dimensions for root window
        self.root.title("CS-320-AI-GUI")
        self.root.geometry("800x600")

        # Function call to create widgets
        self.createWidgets()

    '''
        createWidgets Function: Used to create the different widgets (buttons, labels, dropdown menus, sliders, etc.) for the GUI. 
            Parameters:
                - self: instance of class object
    '''
    def createWidgets(self):
        # Label for dropdown menu
        ttk.Label(self.root, text = "OpenAI Gym Environments:").grid(row = 0, column = 0, padx = 10, pady = 10)

        # List of available gym environments to choose from 
        self.environmentIDs = [
            'Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1',
            'LunarLander-v2', 'CarRacing-v2', 'BipedalWalker-v3',
            'Blackjack-v1', 'Taxi-v3', 'CliffWalking-v0', 'FrozenLake-v1'
        ]

        # Create dropdown menu
        self.selectedEnvironment = tk.StringVar()
        self.environmentDropdown = ttk.Combobox(self.root, textvariable = self.selectedEnvironment, values = self.environmentIDs)
        self.environmentDropdown.grid(row = 0, column = 1, padx = 10, pady = 10)

        # Buttons for loading chosen environment 
        ttk.Button(self.root, text = "Load Environment", command = self.loadEnvironment).grid(row = 1, column = 0, pady = 10)
        
        # Buttons for displaying data plot 
        ttk.Button(self.root, text = 'Plot', command = self.displayPlot).grid(row = 1, column = 1, pady = 10)

        # Function call to embed pygame environment window into GUI 
        self.embedPygame()

    '''
        embedPygame Function: Used to embed pygame window into the GUI. 
            Parameters:
                - self: instance of class object
    '''
    def embedPygame(self):
        # Create frame that will be used to embed the pygame screen
        self.embedFrame = tk.Frame(self.root, width = 800, height = 600)
        self.embedFrame.grid(row = 2, column = 0, columnspan = 2, padx = (120, 10), pady = 10) 

        # Put pygame environment into the frame
        os.environ['SDL_WINDOWID'] = str(self.embedFrame.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        pygame.display.init()
        self.screen = pygame.display.set_mode((800, 600))

    '''
        loadEnvironment Function: Used to embed the pygame screen into the Tkinter GUI frame
            Parameters:
                - self: instance of class object
    '''
    def loadEnvironment(self):
        # Load selected environment
        selectedID = self.selectedEnvironment.get()
        
        # Check if an environment wasn't selected
        if not selectedID:
            self.showError("Choose Environment from Dropdown Menu")
            return

        # Check if the selected environment is valid
        if selectedID not in self.environmentIDs:
            self.showError("Choose a valid environment")
            return

        # Check for errors when creating/loading the chosen environment
        try:
            # Create environment
            self.environment = gym.make(selectedID, render_mode = "human")
            
            # Function call to render the environment
            self.renderEnvironment()

        # Handle errors
        except gym.error.Error as e:
            self.showError(f"Error: Creating Environment: {e}")

    '''
        renderEnvironment Function: Used to render the selected gym environment using pygame. 
            Parameters:
                - self: instance of class object
    '''
    def renderEnvironment(self):
        # Reset and render the environment
        self.state = self.environment.reset() 
        self.environment.render()  
        self.done = False
        self.stepCount = 0

        # Create separate thread for training session
        self.trainingThread = threading.Thread(target = self.trainEnvironment)
        self.trainingThread.start()

    '''
        trainEnvironment Function: Used to train the environment (this uses random actions for now) until the episode ends or a maximum number of steps is met. 
            Parameters:
                - self: instance of class object
    '''
    def trainEnvironment(self):
        maxSteps = 500
        
        # Train until the episode is finished or when the maximum number of steps is met
        while not self.done and self.stepCount < maxSteps:
            # Chooses random action
            action = self.environment.action_space.sample()

            # Performs the random action
            observation, reward, self.done, _, info = self.environment.step(action)
            
            # Updates the display
            pygame.display.flip()

            # Increments the step counter
            self.stepCount += 1

    '''
        displayPlot Function: Used to display data from training sessions using aiPlot class from gymplots file. 
            Parameters:
                - self: instance of class object
    '''
    def displayPlot(self):
        # Check for errors when displaying the plot (using aiPlot)
        try:
            # Create instance of aiPlot class
            plot = aiPlot(step_value = 10, calculate_avg = True)
            
            # Iterate over a range of steps
            for step in range(100):
                # Generate random reward
                reward = np.random.normal(0, 1)
                
                # Update plot with current step/reward
                plot.update(step, reward)

                # Pause for updating the plot to see real-time results
                plt.pause(0.001)

            # Show completed plot after episode finishes or max steps is met
            plot.show()

        # Handle errors
        except Exception as e:
            self.showError(f"Error displaying plots: {e}")

    '''
        showError Function: Used to display error messages in a pop up window to the user. 
            Parameters:
                - self: instance of class object
                - message: error that will be displayed
    '''
    def showError(self, message):
        # Create new top-level window
        errorWindow = tk.Toplevel(self.root)

        # Create a title and label for error window
        errorWindow.title("Error")
        ttk.Label(errorWindow, text = message).pack(padx = 10, pady = 10)

        # Button for closing the error window
        ttk.Button(errorWindow, text = "Close", command = errorWindow.destroy).pack(pady = 10)

'''
    Main Function
'''
if __name__ == "__main__":
    # Create Tkinter winodw
    root = tk.Tk()

    # Initialize GUI
    application = EnvironmentGUI(root)
    
    # Run Tkinter event loop
    root.mainloop()
