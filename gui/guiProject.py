"Imports"
import tkinter as tk
from tkinter import ttk
import pygame
import gym
import os
import threading
import csv

"GUI Class: "
class environmentGUI:
    stepsDefault = 1000 # Initialize default maximum steps

    "Init Function: constructor method for initializing the GUI environment"
    def __init__(self, root, outputPath):
        self.root = root
        self.outputPath = outputPath
        self.setupGUI()
        self.trainingData = []
        self.selectedID = None
        self.stopTraining = threading.Event()
        self.maxSteps = self.stepsDefault
        self.sessionNumber = 1

    "setupGUI Function: method for setting up the GUI window with widgets (labels, button, boxes, etc.)"
    def setupGUI(self):
        # Set title and dimensions for the main root window
        self.root.title("CS-320-AI-GUI")
        self.root.geometry("800x600")

        # 
        ttk.Label(self.root, text = "OpenAI Gym Environments:").grid(row = 0, column = 0, padx = 10, pady = 10, sticky = "w")

        # Available gym environments the user can choose from
        self.environmentIDs = [
            # Classic Control
            'Acrobot-v1', 'CartPole-v1', 'MountainCarContinuous-v0', 'MountainCar-v0', 'Pendulum-v1',
            
            # Box2D
            'LunarLander-v2', 'CarRacing-v2', 'BipedalWalker-v3',
            
            # Toy Text
            'Blackjack-v1', 'Taxi-v3', 'CliffWalking-v0', 'FrozenLake-v1'
        ]

        # Dropbox to display the available environments
        self.selectedEnvironment = tk.StringVar()
        self.environmentDropdown = ttk.Combobox(self.root, textvariable = self.selectedEnvironment, values = self.environmentIDs)
        self.environmentDropdown.grid(row = 0, column = 1, padx = 10, pady = 10, sticky = "w")

        # Entry box for the user to adjust the maximum episode steps 
        ttk.Label(self.root, text = "Max Steps:").grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "w")
        self.maxEpisodeStepsEntry = ttk.Entry(self.root)
        self.maxEpisodeStepsEntry.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "w")

        # Frame to display buttons on the GUI window
        buttonFrame = tk.Frame(self.root)
        buttonFrame.grid(row = 0, column = 2, rowspan = 3, padx = 10, pady = 10, sticky = "ns")

        # Function call to create specific buttons
        self.createButton(buttonFrame, "Load Environment", self.loadEnvironment, 5, 5)
        self.createButton(buttonFrame, "Start", self.startTrainingSession, 5, 5)
        self.createButton(buttonFrame, "Stop", self.stopTraining, 5, 5)
        self.createButton(buttonFrame, "Quit", self.quitEnvironment, 5, 5)

        # Function call to embed the pygame window into the GUI window
        self.embedPygame()

    " createButton Function: method for creating button widgets "
    def createButton(self, parent, text, command, padx, pady):
        button = ttk.Button(parent, text = text, command = command)
        button.pack(fill = "x", padx = padx, pady = pady)
        return button

    " embedPygame Function: method for embedding the pygame display into the GUI window "
    def embedPygame(self):
        # Check if pygame display was initialized
        if not pygame.display.get_init():
            pygame.display.init()

        # Frame for embedding pygame display into
        self.embedFrame = tk.Frame(self.root, width = 500, height = 500)
        self.embedFrame.grid(row = 3, column = 0, columnspan = 3, padx = (120, 10), pady = 10)

        # Put pygame environment display into the frame
        os.environ['SDL_WINDOWID'] = str(self.embedFrame.winfo_id())
        os.environ['SDL_VIDEODRIVER'] = 'windib'
        self.screen = pygame.display.set_mode((800, 600))

    " loadEnvironment Function: method for loading the environment chosen by the user"
    def loadEnvironment(self):
        # Close current environment if it already exists
        if hasattr(self, 'environment') and self.environment:
            self.environment.close() 

        # Get chosen environment
        self.selectedID = self.selectedEnvironment.get()

        # Check if an environment wasn't selected
        if not self.selectedID:
            self.showError("Choose an environment from the dropdown menu")
            return

        # Check if the selected environment is valid
        if self.selectedID not in self.environmentIDs:
            self.showError("Choose a valid environment")
            return

        # Create environment and handle any errors
        try:
            self.environment = gym.make(self.selectedID, render_mode = "human")
            self.resetEnvironment()
        except gym.error.Error as e:
            self.showError(f"Error: Creating Environment: {e}")

    " resetEnvironment Function: method for properly ressetting the environments before creating new ones "
    def resetEnvironment(self):
        self.state = self.environment.reset()
        self.environment.render()
        self.done = False
        self.stepCount = 0

    " startTrainingSession Function: method for starting a training session "
    def startTrainingSession(self):
        # Check if an environment has been loaded properly
        if not self.selectedID:
            self.showError("Need to load an environment")
            return
        
        # Function call to initialize training session with specific settings
        self.initializeTraining()

    " initializeTraining Function: method for setting variables for training sessions "
    def initializeTraining(self):
        self.state = self.environment.reset()
        self.environment.render()
        self.done = False
        self.stepCount = 0
        self.stopTraining.clear()

        # Get the maximum steps chosen by the user
        maxEpisodeSteps = self.maxEpisodeStepsEntry.get()

        # Set the maximum steps for training sessions
        if maxEpisodeSteps.strip():
            try:
                self.maxSteps = int(maxEpisodeSteps)
            except ValueError:
                self.showError("Need to enter a valid integer")
                return
        else:
            self.maxSteps = self.stepsDefault

        # Starts a new thread for executing the training process
        self.trainingThread = threading.Thread(target = self.trainEnvironment)
        self.trainingThread.start()

    " trainEnvironment Function: Used to train the environment (this uses random actions for now) "
    def trainEnvironment(self):
        # Total Reward 
        episodeReward = 0

        # Train until maximum steps is reached, environment is done, or training session stops
        while self.stepCount < self.maxSteps and not self.done and not self.stopTraining.is_set():
            action = self.environment.action_space.sample()
            observation, reward, self.done, _, info = self.environment.step(action)
            episodeReward += reward
            self.trainingData.append((self.selectedID, reward, self.stepCount, episodeReward, len(self.trainingData), self.maxSteps))
            pygame.display.flip()
            self.stepCount += 1

        # Function call to save metrics from training session
        self.saveTrainingData()

    " stopTraining Function: method for pausing the training session once started"
    def stopTraining(self):
        self.stopTraining.set()

    " quitEnvironment Function: Used to quit the GUI "
    def quitEnvironment(self):
        # Checks if environment attribute exists and closes GUI components
        if hasattr(self, 'environment') and self.environment:
            self.stopTraining.set()
            self.trainingThread.join()
            self.saveTrainingData()
            with self.environment:
                pass
        self.root.quit()

    " saveTrainData Function: Used to save the data from training sessions into .csv files "
    def saveTrainingData(self):
        # Get output path and construct path for output.csv files
        outputFile = os.path.join(self.outputPath, f'training_data_session_{self.sessionNumber}.csv')
        fieldnames = ['Library', 'Environment', 'Reward', 'Step', 'Total Reward', 'Max Steps']

         # Open csv file in write mode
        with open(outputFile, mode = 'w', newline = '') as file:
            writer = csv.DictWriter(file, fieldnames = fieldnames)
            writer.writeheader()

            # Iterate over the collected data
            for data in self.trainingData:
                libraryName = self.getLibraryName(data[0])
                environmentName = data[0]
                rowData = {
                    'Library': libraryName,
                    'Environment': environmentName,
                    'Reward': data[1],
                    'Step': data[2],
                    'Total Reward': data[3],
                    'Max Steps': len(self.trainingData),
                }

                # Write data into rows
                writer.writerow(rowData)

        # Increment session counter and clear training data for new sessions
        self.sessionNumber += 1
        self.trainingData = []

    " getLibraryName Function: gets the gym library name corresponding to the selected environment used for training "
    def getLibraryName(self, environmentID):
        # Name of environment library correspinding to their gym environments
        library = {
            "Acrobot": "Classic Control",
            "CartPole": "Classic Control",
            "MountainCar": "Classic Control",
            "Pendulum": "Classic Control",
            "LunarLander": "Box2D",
            "CarRacing": "Box2D",
            "BipedalWalker": "Box2D",
            "Blackjack": "Toy Text",
            "Taxi": "Toy Text",
            "CliffWalking": "Toy Text",
            "FrozenLake": "Toy Text"
        }

        # Iterate over key-value pairs in library dictionary and return name
        for prefix, libraryName in library.items():
            if environmentID.startswith(prefix):
                return libraryName
        return "Unknown"

    " showError Function: displays messages in the form of a pop up window to the user " 
    def showError(self, message):
        errorWindow = tk.Toplevel(self.root)
        errorWindow.title("Error")
        ttk.Label(errorWindow, text = message).pack(padx = 10, pady = 10)
        ttk.Button(errorWindow, text = "Close", command = errorWindow.destroy).pack(pady = 10)

" Main Function "
def main():
    root = tk.Tk()
    outputPath = "output"
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    application = environmentGUI(root, outputPath)
    root.mainloop()

if __name__ == "__main__":
    main()
