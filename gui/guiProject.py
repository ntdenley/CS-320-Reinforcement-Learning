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
import csv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import ttk
from gymplots import aiPlot

# Initialize pygame display
pygame.init()

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
                - outputPath: 
    '''
    def __init__(self, root, outputPath):
        self.root = root
        self.outputPath = outputPath
        self.setupGUI()
        self.trainingData = []
        self.selectedID = None
        self.stopTraining = False  # Flag to signal the training thread to stop
        self.maxSteps = 1000  # Default value for max steps
        self.sessionNumber = 1  # Initialize traning session counter

    '''
        setupGUI Function: Used to set up the graphical user interface (GUI) by providing the dimensions/title for the Tkinter root window. 
            Parameters:
                - self: instance of class object
    '''
    def setupGUI(self):
        # Set title and dimensions for root window
        self.root.title("CS-320-AI-GUI")
        self.root.geometry("800x600")

        # Function call to create widgets and embed pygame display
        self.createWidgets()
        self.embedPygame()

    '''
        createWidgets Function: Used to create the different widgets (buttons, labels, dropdown menus, sliders, etc.) for the GUI. 
            Parameters:
                - self: instance of class object
    '''
    def createWidgets(self):
        ttk.Label(self.root, text = "OpenAI Gym Environments:").grid(row = 0, column = 0, padx = 10, pady = 10, sticky = "w")

        self.environmentIDs = [
            # Classic Control Environments
            'Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1',
            
            # Box2D Environments
            'LunarLander-v2', 'CarRacing-v2', 'BipedalWalker-v3',
            
            # Toy Text Environments
            'Blackjack-v1', 'Taxi-v3', 'CliffWalking-v0', 'FrozenLake-v1'
        ]

        # Create Dropbox to show environments that the user can choose from
        self.selectedEnvironment = tk.StringVar()
        self.environmentDropdown = ttk.Combobox(self.root, textvariable = self.selectedEnvironment, values = self.environmentIDs)
        self.environmentDropdown.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = "w")

        # Create entry box for user to adjust the maximum episode steps for training sessions
        ttk.Label(self.root, text="Max Episode Steps:").grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "w")
        self.maxEpisodeStepsEntry = ttk.Entry(self.root)
        self.maxEpisodeStepsEntry.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "w")

        # Adjusts the buttons layout
        buttonFrame = tk.Frame(self.root)
        buttonFrame.grid(row = 0, column = 2, rowspan = 3, padx = 10, pady = 10, sticky = "ns")

        # Create Buttons
        ttk.Button(buttonFrame, text = "Load Environment", command = self.loadEnvironment).pack(fill = "x", padx = 5, pady = 5)
        ttk.Button(buttonFrame, text ='Start Training Session', command = self.startTrainingSession).pack(fill = "x", padx = 5, pady = 5)
        ttk.Button(buttonFrame, text = 'Quit', command = self.quitEnvironment).pack(fill = "x", padx = 5, pady = 5)

    '''
        embedPygame Function: Used to embed pygame window into the GUI. 
            Parameters:
                - self: instance of class object
    '''
    def embedPygame(self):
        # Check if the pygame display was initialized
        if not pygame.display.get_init():
            pygame.display.init()
        
        # Create from to embed pygame display into
        self.embedFrame = tk.Frame(self.root, width=600, height=500)
        self.embedFrame.grid(row=3, column=0, columnspan=3, padx=(120, 10), pady=10)
        
        # Put pygame environment display into the frame
        os.environ['SDL_WINDOWID'] = str(self.embedFrame.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        self.screen = pygame.display.set_mode((800, 600))

    '''
        loadEnvironment Function: Used to embed the pygame screen into the Tkinter GUI frame
            Parameters:
                - self: instance of class object
    '''
    def loadEnvironment(self):
        # Load selected environment
        self.selectedID = self.selectedEnvironment.get()

        # Check if an environment wasn't selected
        if not self.selectedID:
            self.showError("Choose Environment from Dropdown Menu")
            return
        
        # Check if the selected environment is valid
        if self.selectedID not in self.environmentIDs:
            self.showError("Choose a valid environment")
            return

        # Check for errors when creating/loading the chosen environment
        try:
            # Create environment
            self.environment = gym.make(self.selectedID, render_mode = "human")
        except gym.error.Error as e:
            # Handle errors
            self.showError(f"Error: Creating Environment: {e}")

    '''
        startTrainingSession Function: Used to 
            Parameters:
                - self: instance of class object
    '''
    def startTrainingSession(self):
        # Check if an environment has been loaded
        if not self.selectedID:
            self.showError("Load an environment first")
            return

        # Function call to render the chosen environment for training session
        self.renderEnvironment()

    '''
        renderEnvironment Function: Used to render the selected gym environment using pygame. 
            Parameters:
                - self: instance of class object
    '''
    def renderEnvironment(self):
        self.state = self.environment.reset()
        self.environment.render()
        self.done = False
        self.stepCount = 0
        self.stopTraining = False  # Reset flag
        self.maxSteps = int(self.maxEpisodeStepsEntry.get())  # Get user input for max episode steps
        self.trainingThread = threading.Thread(target = self.trainEnvironment)
        self.trainingThread.start()

    '''
        trainEnvironment Function: Used to train the environment (this uses random actions for now) until the episode ends or a maximum number of steps is met. 
            Parameters:
                - self: instance of class object
    '''
    def trainEnvironment(self):
        # Total reward from training session
        episodeReward = 0

        # Train until maximum steps is reached, environment signals its done, or training session is stopped
        while self.stepCount < self.maxSteps and not self.done and not self.stopTraining:
            action = self.environment.action_space.sample()
            observation, reward, self.done, _, info = self.environment.step(action)
            episodeReward += reward
            self.trainingData.append((self.selectedID, reward, self.stepCount, episodeReward, len(self.trainingData), self.maxSteps))
            pygame.display.flip()
            self.stepCount += 1

        # Function call to save the data from training session into .csv file
        self.saveTrainingData() 

    '''
        quitEnvironment Function: Used to quit the GUI besides closing the window. 
            Parameters:
                - self: instance of class object
    '''
    def quitEnvironment(self):
        # Checks if environment attribute exists and closes GUI components
        if hasattr(self, 'environment') and self.environment:
            self.stopTraining = True  
            self.trainingThread.join()
            self.environment.close()    
        self.root.quit()

    '''
        saveTrainData Function: Used to save the data from training sessions into .csv files. 
            Parameters:
                - self: instance of class object
    '''
    def saveTrainingData(self):
        # Get output path and construct path for output.csv files
        outputPath = self.outputPath
        outputFile = os.path.join(outputPath, f'training_data_session_{self.sessionNumber}.csv')

        # Open csv file in write mode using a context manager
        with open(outputFile, mode = 'w', newline = '') as file:
            # Create csv writer object
            writer = csv.writer(file)
            
            # Write column names
            writer.writerow(['Library Name', 'Environment Name', 'Reward Per Step', 'Step', 'Total Reward', 'Total Steps', 'Max Steps'])

            # Iterate over collected data collected
            for data in self.trainingData:
                # Get the gym environment library name
                libraryName = self.getLibraryName(data[0])
                
                # Get environment name
                environmentName = data[0]
                
                # Write data into rows
                writer.writerow([libraryName, environmentName] + list(data[1:]))

        # Incrementsession counter and clear training data for new sessions
        self.sessionNumber += 1
        self.trainingData = []

    '''
        getLibraryName Function: Used to get the gym library name corresponding to environments. 
            Parameters:
                - self: instance of class object
                - environmentID: the environment selected for training
    '''
    def getLibraryName(self, environmentID):
        if environmentID.startswith("Acrobot") or environmentID.startswith("CartPole") or environmentID.startswith("MountainCar") or environmentID.startswith("Pendulum"):
            return "Classic Control"
        elif environmentID.startswith("LunarLander") or environmentID.startswith("CarRacing") or environmentID.startswith("BipedalWalker"):
            return "Box2D"
        elif environmentID.startswith("Blackjack") or environmentID.startswith("Taxi") or environmentID.startswith("CliffWalking") or environmentID.startswith("FrozenLake"):
            return "Toy Text"
        else:
            return "Unknown"

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
def main():
    # Create Tkinter winodw
    root = tk.Tk()

    # Define output path and check if it exists. If not, create the directory
    outputPath = "output"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    # Initialize GUI
    application = EnvironmentGUI(root, outputPath)
    
    # Run Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
