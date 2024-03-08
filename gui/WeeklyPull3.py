'''
    Resource(s):
        https://pypi.org/project/pillow/
        https://gymnasium.farama.org/
        https://sourceforge.net/projects/swig/
'''

#   Imports and Declarations
import gymnasium as gym                 #   OpenAI Gym Environments
import numpy as np
import tkinter as tk                    #   GUI
from tkinter import ttk
from PIL import Image, ImageTk          #   Image Processing

#   GUI Class Definition
class environmentGUI:
    def __init__(self, root):

        #   Initializes root window (top level and/or main window)
        self.root = root
        self.root.title("CS-320-AI-GUI")

        #   Creates label for the dropdown menu
        self.environmentLabel = ttk.Label(root, text = "OpenAI Gym Environments:")
        self.environmentLabel.grid(row = 0, column = 0, padx = 10, pady = 10)

        #   Lists available OpenAI gymnasium environments
        self.environmentIDs = [
            'LunarLander-v2',
        ]

        #   Variable to store selected environment
        self.selectedEnvironment = tk.StringVar()

        #   Creates dropdown menu
        self.environmentDropdown = ttk.Combobox(root, textvariable = self.selectedEnvironment, values = self.environmentIDs)
        self.environmentDropdown.grid(row = 0, column = 1, padx = 10, pady = 10)

        #   Creates button to load selected environment
        self.loadButton = ttk.Button(root, text = "Load Environment", command = self.loadEnvironment)
        self.loadButton.grid(row = 1, column = 0, columnspan = 2, pady = 10)

        #   Creates canvas to display rendered environment
        self.canvas = tk.Canvas(root, width = 400, height = 300)
        self.canvas.grid(row = 2, column = 0, columnspan = 2, pady = 10)

    def loadEnvironment(self):
        #   Checks to see if an environment is selcted from the dropdown menu
        if not self.selectedEnvironment.get():
            self.errorMessage("Choose Environment from Dropdown Menu")
            return
        
        #   Creates selected Gym environment 
        environmentID = self.selectedEnvironment.get()     

        try:
            #   Tries creating the chosen environmnet
            environment = gym.make(environmentID)  
        except gym.error.Error as e:
            #   Handles errors if one occurrs
            self.errorMessage(f"ERROR: Creating Environment: {e}")
            return
        
        #   Environment was successfully created (tuple, boolean)
        state, done = environment.reset()

        #    Renders the environment and displays it on a canvas
        image = Image.fromarray(state)
        image = image.resize((400, 300), resample = 3)
        imageTK = ImageTk.PhotoImage(image)
        self.canvas.imageTK = imageTK  
        self.canvas.create_image(0, 0, anchor = tk.NW, image = imageTK)

    def errorMessage(self, message):
        #   Creates new top level window to display the error message over the root window
        errorWindow = tk.Toplevel(self.root)
        errorWindow.title("ERROR")
        
        #   Label to display the error message
        errorLabel = ttk.Label(errorWindow, text = message)
        errorLabel.pack(padx = 10, pady = 10)

        #   Creates a button to close the error window
        closeButton = ttk.Button(errorWindow, text = "Close", command = errorWindow.destroy)
        closeButton.pack(pady = 10)

#   Main Function
if __name__ == "__main__":
    #   Creates main Tkinter window, instance of class, and loops
    root = tk.Tk()  
    app = environmentGUI(root)
    root.mainloop()
