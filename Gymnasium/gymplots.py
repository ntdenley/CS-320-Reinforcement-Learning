"""
gymplots.py - handles plotting based on new plot info
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

plt.ion()

class aiPlot():
    """
    Args:
        step_value: how often the graph should update. Every call to
        update() advances the frame, and the plot is updated every Nth
        frame, where n is the step_value.

        calculate_avg: should the total average be calculated when updating
        the plot? (Default -> False)
    """

    def __init__(self, step_value, calculate_avg=False):

        # Initialize plot data arrays
        self.x = []
        self.y = []
        self.avgs = []

        # Assign vital class members
        self.step_value = step_value
        self.frame = 0
        self.graph = None
        self.calc_avg = calculate_avg
        
    def update(self, xVal, yVal):

        # Check if plot will be updated
        if self.frame % self.step_value == 0:

            # Add values to plot data
            self.x.append(xVal)
            self.y.append(yVal)

            # Remove graph to update (if it exists)
            if self.graph is not None:
                self.graph.remove()
            
            # Create new graph with updated plot data
            self.graph = plt.plot(self.x, self.y, color='g')[0] # default color to green (for now)
            
            # Check if we care for the average
            if self.calc_avg:
                # Calculate the average, append it to list of averages, and add second plot line.
                average = sum(self.y) / len(self.y)
                self.avgs.append(average)
                plt.plot(self.x, self.avgs, color='r')[0] # default color to red 

            # Dynamically size the plot based on x.
            plt.xlim(self.x[0], self.x[-1])
            
            # Pause to prevent plot update overlapping
            plt.pause(0.001)
        
        # Increment frame counter
        self.frame += 1