'''
Imports & Declarations
- Gym (Gymnasium): OpenAI Environment Library
- Pygame: Multi Media Game Creation Library
- Numpy: Numerical Operation Library
- tkinter (tk, ttk): GUI Library
- Pillow: Rendering Library
'''
import gym
import pygame
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from gymplots import aiPlot  # Import aiPlot class from gymplots.py

'''
    GUI Class:
        Initalizes and handles aspects of the main Tkinter GUI window and the OpenAI environments such as 
        the available environment IDs, GUI widgets (labels, buttons, and canvas), resizing, refreshing, 
        loading/rendering, etc.  
'''
class environmentGUI:
    '''
        Function for initializing the GUI class and setting up the Tkinter root window.
    '''
    def __init__(self, root):
        #   Root window (top level/main window)
        self.root = root
        self.root.title("CS-320-AI-GUI")

        #   Dropdown menu label
        self.environmentLabel = ttk.Label(root, text = "OpenAI Gym Environments:")
        self.environmentLabel.grid(row = 0, column = 0, padx = 10, pady = 10)

        #   Available OpenAI gym environments (will add more in the future)
        self.environmentIDs = [
            #   Classic Control
            'Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1',
            #   Box2D
            'LunarLander-v2', 'CarRacing-v2', 'BipedalWalker-v3',
            #   Toy Text
            'Blackjack-v1', 'Taxi-v3', 'CliffWalking-v0', 'FrozenLake-v1'
        ]

        self.maxSteps = 200

        #   String variable to store selected environment
        self.selectedEnvironment = tk.StringVar()

        #   Dropdown menu
        self.environmentDropdown = ttk.Combobox(root, textvariable = self.selectedEnvironment, values = self.environmentIDs)
        self.environmentDropdown.grid(row = 0, column = 1, padx = 10, pady = 10)

        #   Button to load selected environment
        self.loadButton = ttk.Button(root, text = "Load Environment", command = self.loadEnvironment)
        self.loadButton.grid(row = 1, column = 0, pady = 10)

        #   Button to display gymplots
        self.plotButton = ttk.Button(root, text = 'Plot', command = self.displayPlots)
        self.plotButton.grid(row = 1, column = 1, columnspan = 1, pady = 10)

        #   Initializes canvas size based on GUI size
        initialWidth = 800
        initialHeight = 600
        self.canvas = tk.Canvas(root, width = initialWidth, height = initialHeight)
        self.canvas.grid(row = 2, column = 0, columnspan = 2, pady = 10)

        #   Binds the configure root window event to resizeWindow function
        root.bind('<Configure>', self.resizeWindow)


    '''
        Function for dynamically updating the canvas size based off the calculated window size. 
    '''
    def resizeWindow(self, event):
        updateWidth = event.width - 150
        updateHeight = event.height - 150
        self.canvas.config(width = updateWidth, height = updateHeight)

    '''
        Function for rendering the chosen gym environment.
    '''
    def renderEnvironment(self, environment, maxSteps):
        #   Sets up rendering window
        environment.render()

        #   Gets initial state
        state = environment.reset()
        done = False
        stepCounter = 0

        while not done and stepCounter < maxSteps:
            action = environment.action_space.sample()
            observation, reward, done, _, info = environment.step(action)

            # Add the following lines to update the plot
            self.plot.update(stepCounter, reward)  # You might need to adjust the arguments based on your data
            plt.pause(0.001)  # Ensure the plot updates in real-time

            environment.render()
            stepCounter += 1
        pygame.quit()

    '''
        Function for loading the chosen environment. 
    '''
    def loadEnvironment(self):
        #   Checks if an environment has been selected from the dropdown menu
        if not self.selectedEnvironment.get():
            self.errorMessage("Choose Environment from Dropdown Menu")
            return

        #   Gets selected Gym environment
        environmentID = self.selectedEnvironment.get()

        #   Checks if selected environment is available
        if environmentID not in self.environmentIDs:
            self.errorMessage("Invalid Environment Selection")
            return

        #   Attempt to create the selected environment
        try:
            environment = gym.make(environmentID, render_mode="human")
        except gym.error.Error as e:
            self.errorMessage(f"Error: Creating Environment: {e}")
            return

        #   Renders a new version of the selected environment
        environment.reset()
        self.renderEnvironment(environment, self.maxSteps)

    '''
        Function for displaying the plots from gymplots.py
    '''
    def displayPlots(self):
        #   Create instance of AI Plot
        self.plot = aiPlot(step_value= 10, calculate_avg= True)

        #   Adds data to plot
        for step in range(100):
            reward = np.random.normal(0, 1)
            self.plot.update(step, reward)
            plt.pause(0.001)

        #   Display Plot
        self.plot.show()

    '''
        Function for prompting the user when errors occur.
    '''
    def errorMessage(self, message):
        #   New top level window to display error message over root window
        errorWindow = tk.Toplevel(self.root)
        errorWindow.title("Uh Oh")

        #   Label to display error message
        errorLabel = ttk.Label(errorWindow, text=message)
        errorLabel.pack(padx=10, pady=10)

        #   Button to close error window
        closeButton = ttk.Button(errorWindow, text="Close", command=errorWindow.destroy)
        closeButton.pack(pady=10)

'''
    Main Function
'''
if __name__ == "__main__":
    #   Creates main Tkinter window, instance of class, and loops
    root = tk.Tk()
    app = environmentGUI(root)
    root.mainloop()
