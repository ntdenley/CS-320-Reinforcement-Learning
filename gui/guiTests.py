'''
Source(s):
    - https://docs.python.org/3/library/unittest.html
    - https://docs.python.org/3/library/unittest.mock.html
'''

# Imports
import unittest
import gym
import numpy as np
import tkinter as tk
from guiProject import environmentGUI 

class TestGui(unittest.TestCase):
# ------------------------------------------------------------------------------------
# Black-box Tests (Acceptance): Thoroughly test requirements or features from cool-cam
# ------------------------------------------------------------------------------------
    
    '''
        # 1 - Tests whether or not the environment IDs are loading successfully
    '''
    def test_loadingEnvironments(self):
        outputPath = "output"
        thing = environmentGUI(tk.Tk())
        self.assertEqual(thing.environmentIDs, 
                         ['Acrobot-v1', 
                          'CartPole-v1', 
                          'MountainCarContinuous-v0', 
                          'MountainCar-v0', 
                          'Pendulum-v1', 
                          'LunarLander-v2', 
                          'CarRacing-v2', 
                          'BipedalWalker-v3', 
                          'Blackjack-v1', 
                          'Taxi-v3', 
                          'CliffWalking-v0', 
                          'FrozenLake-v1'
                          ])

    '''
        # 2 - Tests whether or not error messages are being displayed successfully when an invalid environment is chosen
    '''
    def test_invalidEnvironment(self):
        outputPath = "output"
        thing = environmentGUI(tk.Tk(), outputPath)
        with self.assertRaises(gym.error.Error):
            thing.loadEnvironment()

    '''
        # 3 - Tests whether or not the environments are rendering successfully
    '''
    def test_renderEnvironment(self):
        outputPath = "output"
        thing = environmentGUI(tk.Tk(), outputPath)
        

        # NEED TO FIX
    
    '''
        # 4 - Tests whether or not the dropdown menu is initialized successfully
    '''
    def test_initializeDropdown(self):
        outputPath = "output"
        thing = environmentGUI(tk.Tk(), outputPath)
        dropdown_values = list(thing.environmentDropdown['values'])  # Convert tuple to list
        self.assertEqual(dropdown_values, thing.environmentIDs)

# ------------------------------------------------------------------
# White-box Tests: Provide coverage of functions/procedures/methods
# ------------------------------------------------------------------        
    
    '''
        # 5 - Verifies that the errorMessage function is called/executes without errors (statement coverage)
    '''
    '''
        def errorMessage(self, message):
            errorWindow = tk.Toplevel(self.root)
            errorWindow.title("Uh Oh")
            errorLabel = ttk.Label(errorWindow, text = message)
            errorLabel.pack(padx = 10, pady = 10)
            closeButton = ttk.Button(errorWindow, text = "Close", command = errorWindow.destroy)
            closeButton.pack(pady = 10)
    '''
    def test_errorMessage_coverage(self):
        thing = environmentGUI(tk.Tk())
        thing.errorMessage("Test Error Message")

    '''
        # 6 - Verifies that the resizeWindow function is called/executed without having any errors, assumes window resize event occured. (statement coverage)
    '''
    '''
        def resizeWindow(self, event):
            updateWidth = event.width - 150
            updateHeight = event.height - 150
            self.canvas.config(width=updateWidth, height=updateHeight)            
    '''

    def test_resizeWindow_coverage(self):
        thing = environmentGUI(tk.Tk())
        event = tk.Event()
        thing.resizeWindow(event) 

    '''
        # 7 - Verifies that the renderEnvironment function is called/executed without having any errors and
                covers rendering loop, rendering for a specified number of steps, and updating the environment. (statement coverage)
    '''
    '''
                
        def renderEnvironment(self, environment, maxSteps):
            environment.render()
            state = environment.reset()
                done = False
                stepCounter = 0
                while not done and stepCounter < maxSteps:
                    action = environment.action_space.sample()
                    observation, reward, done, _, info = environment.step(action)
                    self.plot.update(stepCounter, reward) 
                    plt.pause(0.001)
                    environment.render()
                    stepCounter += 1                    
                pygame.quit()
    '''
    
    def test_renderEnvironment_coverage(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make') as mock:
            mockEnv = mock.returnValue
            mockEnv.reset.returnValue = 0
            mockEnv.step.returnValue = (0, 0, False, {}, {}) 
            thing.renderEnvironment(mockEnv, 200)

    '''
        # 8 - Verifies that the loadEnvironemt function is called/executed without having any errors and 
                covers the loading, error handling, and rendering of the environments, (statement coverage)
    '''
    '''
        def loadEnvironment(self):
            if not self.selectedEnvironment.get():
                self.errorMessage("Choose Environment from Dropdown Menu")
                return
            environmentID = self.selectedEnvironment.get()
            if environmentID not in self.environmentIDs:
                self.errorMessage("Invalid Environment Selection")
                return
            try:
                environment = gym.make(environmentID, render_mode = "human")
            except gym.error.Error as e:
                self.errorMessage(f"Error: Creating Environment: {e}")
                return
            environment.reset()
            self.renderEnvironment(environment, self.maxSteps)
    '''
    def test_loadEnvironment_coverage(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make') as mock:
            mockEnv = mock.returnValue
            mockEnv.reset.returnValue = 0  
            mockEnv.step.returnValue = (0, 0, False, {}, {})  
            thing.loadEnvironment()

# -----------------------------------------------------
# Integration Tests: Thoroughly tests two units at once  
# -----------------------------------------------------
    
    '''
        # 9 - Tests loading the environment and rendering it at once

            - loadEnvironment function (unit: environmentGUI class)
            - renderEnvironment function (unit: environmentGUI class)

            Approach:
                Bottom-up integration test where it first tests loadEnvironment function and verifies its 
                    interaction with renderEnvironment function.
    '''
    def test_loadEnvironment_renderEnvironment(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make') as mock:
            mockEnv = mock.returnValue
            mockEnv.reset.returnValue = 0  
            mockEnv.step.returnValue = (0, 0, False, {}, {})  
            thing.loadEnvironment()
            self.assertTrue(mockEnv.render.called)

    ''' 
        # 10 - Tests loading the environment and displaying plots at once

            - loadEnvironment function (unit: environmentGUI class)
            - displayPlots function (unit: environmentGUI class)
            - aiPlot class (unit: gymplots module, Noah Denley's file)
        
            Approach:
                Bottom-up integration test where it first tests loadEnvironment function, verifies its 
                    interaction with the displayPlots function, and lastly, checks if aiPlot is called properly.
    '''
    def test_loadEnvironment_displayPlots(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make') as mock:
            mockEnv = mock.returnValue
            mockEnv.reset.returnValue = 0  
            mockEnv.step.returnValue = (0, 0, False, {}, {}) 
            with patch('gymplots.aiPlot') as mock_aiplot:
                thing.loadEnvironment()
                thing.displayPlots()
                self.assertTrue(mock_aiplot.returnValue.show.called)
    '''
        # 11 - Tests loading the environment and checking the canvas size at once

             - loadEnvironment function (unit: environmentGUI class)

            Approach:
                Bottom-up integration test where it tests loadEnvironment function and 
                    checks how it impacts the size of the canvas.
    '''
    def test_loadEnvironment_canvasSize(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make') as mock:
            mockEnv = mock.returnValue
            mockEnv.reset.returnValue = 0  
            mockEnv.step.returnValue = (0, 0, False, {}, {}) 
            thing.loadEnvironment()
            self.assertEqual(thing.canvas.winfo_width(), 800)
            self.assertEqual(thing.canvas.winfo_height(), 600)

    '''
        # 12 - Tests loading the environment and displaying an error message at once

             - loadEnvironment function (unit: environmentGUI class)

            Approach:
                Bottom-up integration test where it tests loadEnvironment function and 
                    checks how it's interactting with handling error messages.
    '''
    def test_loadEnvironment_errorMessage(self):
        thing = environmentGUI(tk.Tk())
        with patch('gym.make', side_effect = gym.error.Error('Environment Creation Failed')):
            with patch('tkinter.Toplevel') as mockToplevel:
                thing.loadEnvironment()
                mockToplevel.assert_called_with(thing.root)
    
if __name__ == '__main__':
    unittest.main()
