"""
gymplots.py - handles plotting based on new plot info
"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

plt.ion()

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white', 
          'gray', 'brown', 'orange', 'purple', 'pink', 'olive', 'teal', 'maroon', 'navy']

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
        self.lines = []
        for i in range(10):
            self.lines.append({
                "x": [],
                "y": []
            })

        # Assign vital class members
        self.step_value = step_value
        self.frame = 0
        self.graph = None
        self.calc_avg = calculate_avg
        
    def update(self, xVal, yVal, plot_count):

        # Check if plot will be updated
        if self.frame % self.step_value == 0:

            # Add values to plot data
            self.lines[plot_count-1]["x"].append(xVal)
            self.lines[plot_count-1]["y"].append(yVal)

            # Remove graph to update (if it exists)
            if self.graph is not None:
                self.graph.remove()
            
            # Create new graph with updated plot data
            for i in range(plot_count):
                self.graph = plt.plot(
                    self.lines[i]["x"],
                    self.lines[i]["y"], 
                    label=f'Trial #{i+1}', 
                    color=colors[i]
                )[0]
            
            # # Check if we care for the average
            # if self.calc_avg:
            #     # Calculate the average, append it to list of averages, and add second plot line.
            #     average = sum(self.y) / len(self.y)
            #     self.avgs.append(average)
            #     plt.plot(self.x, self.avgs, color='r')[0] # default color to red 

            # Dynamically size the plot based on x.
            plt.xlim(self.lines[plot_count-1]["x"][0], self.lines[plot_count-1]["x"][-1])
            
            # Pause to prevent plot update overlapping
            plt.pause(0.001)
        
        # Increment frame counter
        self.frame += 1
    
    def close(self):
        plt.close()