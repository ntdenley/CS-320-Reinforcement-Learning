'''
    Package              Version
    -------------------- -------
    cloudpickle          3.0.0
    Farama-Notifications 0.0.4
    gym                  0.26.2
    gym-notices          0.0.8
    gymnasium            0.29.1
    numpy                1.26.3
    pillow               10.2.0
    pip                  23.3.2
    setuptools           69.0.3
    tk                   0.1.0
    typing_extensions    4.9.0
    wheel                0.38.4

    Resource(s):
        https://pypi.org/project/pillow/
        https://gymnasium.farama.org/

'''

#   Imports and Declarations
import gym                              #   OpenAI Gym Environments
import tkinter as tk                    #   GUI
from tkinter import ttk
from PIL import Image, ImageTk          #   Image Processing

#   GUI Class Definition
class environmentGUI:
    def __init__(self, root):

        #   Initializes root window (top level and/or main window)
        self.root = root
        self.root.title("CS-320-AI-GUI")

        #   Creates a label for the dropdown menu
        self.environmentLabel = ttk.Label(root, text="OpenAI Gym Environments:")
        self.environmentLabel.grid(row=0, column=0, padx=10, pady=10)

        #   Lists available environment options
        self.environmentIDs = [
            'LunarLander-v2',
        ]

        #   Variable to store selected environment
        self.selectedEnvironment = tk.StringVar()

        #   Creates dropdown menu
        self.environmentDropdown = ttk.Combobox(root, textvariable=self.selectedEnvironment, values=self.environmentIDs)
        self.environmentDropdown.grid(row=0, column=1, padx=10, pady=10)

        #   Creates button to load the environment selected
        self.loadButton = ttk.Button(root, text="Load Environment", command=self.loadEnvironment)
        self.loadButton.grid(row=1, column=0, columnspan=2, pady=10)

        #   Creates canvas to display rendered environment
        self.canvas = tk.Canvas(root, width=400, height=300)
        self.canvas.grid(row=2, column=0, columnspan=2, pady=10)

    def loadEnvironment(self):
        #   Creates selected Gym environment 
        environmentID = self.selectedEnvironment.get()     
        environment = gym.make(environmentID)             
        state = environment.reset()                         

        #   Renders the environment and displays it on a canvas
        image = Image.fromarray(state)
        image = image.resize((400, 300), Image.ANTIALIAS)
        imageTK = ImageTk.PhotoImage(image)
        self.canvas.imageTK = imageTK  
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imageTK)

#   Main
if __name__ == "__main__":
    #   Creates main Tkinter window, instance of class, and loops
    root = tk.Tk()  
    app = environmentGUI(root)
    root.mainloop()
