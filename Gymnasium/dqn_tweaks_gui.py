import tkinter as tk
import tkinter.ttk as ttk
import torch
import cartpole_dqn as sim
from gymplots import aiPlot

class DQNAdjuster:
    def __init__(self, master=None):

        self.plot = None
        self.plot_count = 0

        # build ui
        self.toplevel1 = tk.Tk() if master is None else tk.Toplevel(master)
        self.toplevel1.configure(background="#5f7c8b", height=200, width=200)
        self.title_frame = tk.Frame(self.toplevel1, name="title_frame")
        self.title_frame.configure(background="#5f7c8b", height=200, width=200)
        self.title = tk.Label(self.title_frame, name="title")
        self.title.configure(
            background="#5f7c8b",
            font="{@Malgun Gothic} 16 {}",
            text='DQN Trainer')
        self.title.grid(column=0, pady=10, row=0)
        self.title_frame.grid(column=0, row=0)
        self.agent_options_frame = ttk.Labelframe(
            self.toplevel1, name="agent_options_frame")
        self.agent_options_frame.configure(
            height=200, text='Agent Options', width=400)
        self.n_training_episodes_slider = tk.Scale(
            self.agent_options_frame, name="n_training_episodes_slider")
        self.N_EPISODES = tk.IntVar()
        self.n_training_episodes_slider.configure(
            from_=1,
            label='Number of Training Episodes:',
            length=300,
            orient="horizontal",
            resolution=1,
            to=1000,
            variable=self.N_EPISODES,
            width=10)
        self.n_training_episodes_slider.grid(column=0, row=0)
        self.n_training_episodes_slider.set(600 if torch.cuda.is_available() else 100)
        self.n_display_episodes_slider = tk.Scale(
            self.agent_options_frame, name="n_display_episodes_slider")
        self.N_EPISODES_DISPLAYED = tk.IntVar()
        self.n_display_episodes_slider.configure(
            from_=0,
            label='Number of Displayed Episodes:',
            length=300,
            orient="horizontal",
            to=25,
            variable=self.N_EPISODES_DISPLAYED,
            width=10)
        self.n_display_episodes_slider.grid(column=0, row=1)
        self.n_display_episodes_slider.set(25)
        self.lr_label = tk.Label(self.agent_options_frame, name="lr_label")
        self.lr_label.configure(
            background="#d7dfe3",
            text='Learning Rate (0.1-0.00001)',
            width=35)
        self.lr_label.grid(column=0, row=2)
        self.lr_entry = tk.Entry(self.agent_options_frame, name="lr_entry")
        self.LR = tk.DoubleVar(value=0.0001)
        self.lr_entry.configure(textvariable=self.LR)
        _text_ = '0.0001'
        self.lr_entry.delete("0", "end")
        self.lr_entry.insert("0", _text_)
        self.lr_entry.grid(column=0, pady=5, row=3)
        self.discount_slider = tk.Scale(
            self.agent_options_frame,
            name="discount_slider")
        self.GAMMA = tk.DoubleVar()
        self.discount_slider.configure(
            from_=0.5,
            label='Discount Factor:',
            length=300,
            orient="horizontal",
            resolution=0.001,
            to=1,
            variable=self.GAMMA,
            width=10)
        self.discount_slider.grid(column=0, row=4)
        self.discount_slider.set(0.99)
        self.batch_size_slider = tk.Scale(
            self.agent_options_frame,
            name="batch_size_slider")
        self.BATCH_SIZE = tk.IntVar()
        self.batch_size_slider.configure(
            from_=1,
            label='Batch Size:',
            length=300,
            orient="horizontal",
            to=256,
            variable=self.BATCH_SIZE,
            width=10)
        self.batch_size_slider.grid(column=0, row=5)
        self.batch_size_slider.set(128)
        self.random_slider = tk.Scale(
            self.agent_options_frame,
            name="random_slider")
        self.EPS_START = tk.DoubleVar()
        self.random_slider.configure(
            from_=0,
            label='Random Factor:',
            length=300,
            orient="horizontal",
            resolution=0.01,
            to=1,
            variable=self.EPS_START,
            width=10)
        self.random_slider.grid(column=0, row=6)
        self.random_slider.set(0.9)
        self.random_decay_slider = tk.Scale(
            self.agent_options_frame, name="random_decay_slider")
        self.EPS_DECAY = tk.IntVar()
        self.random_decay_slider.configure(
            from_=10,
            label='Random Decay',
            length=300,
            orient="horizontal",
            resolution=10,
            to=2000,
            variable=self.EPS_DECAY,
            width=10)
        self.random_decay_slider.grid(column=0, row=7)
        self.random_decay_slider.set(1000)
        self.agent_options_frame.grid(column=0, padx=20, pady=0, row=1)
        self.plot_options_frame = ttk.Labelframe(
            self.toplevel1, name="plot_options_frame")
        self.plot_options_frame.configure(
            height=200, text='Plot Options', width=300)
        self.enable_avg_toggle = ttk.Checkbutton(
            self.plot_options_frame, name="enable_avg_toggle")
        self.show_avg = tk.BooleanVar()
        self.enable_avg_toggle.configure(
            offvalue=False,
            onvalue=True,
            text='Display Average',
            variable=self.show_avg)
        self.enable_avg_toggle.grid(column=0, padx=20, row=0)
        self.save_plot = tk.BooleanVar()
        self.save_plot_toggle = ttk.Checkbutton(
            self.plot_options_frame, name="save_plot_toggle")
        self.save_plot_toggle.configure(
            offvalue=False,
            onvalue=True,
            text='Save Previous Plots',
            variable=self.save_plot)
        self.save_plot_toggle.grid(column=1, padx=20, row=0)
        self.plot_options_frame.grid(column=0, padx=10, pady=10, row=2)
        self.misc_options_frame = ttk.Labelframe(
            self.toplevel1, name="misc_options_frame")
        self.misc_options_frame.configure(
            height=200, takefocus=False, text='Misc. Options', width=300)
        self.cuda_enabled_toggle = ttk.Checkbutton(
            self.misc_options_frame, name="cuda_enabled_toggle", state="enabled" if torch.cuda.is_available() else "disabled")
        self.cuda_enabled_toggle.configure(text='Enable CUDA')
        self.cuda_enabled_toggle.grid(column=0, padx=107, row=0)
        self.misc_options_frame.grid(column=0, padx=10, row=3)
        self.button_frame = ttk.Frame(self.toplevel1, name="button_frame")
        self.button_frame.configure(height=200, width=200)
        button2 = ttk.Button(self.button_frame)
        button2.configure(text='Begin Training!')
        button2.grid(column=0, row=0)
        button2.configure(command=self.begin_training)
        self.button_frame.grid(column=0, pady=20, row=4)

        # Main widget
        self.mainwindow = self.toplevel1

    def run(self):
        self.mainwindow.mainloop()

    def begin_training(self):
        
        if not self.save_plot.get() or not self.plot:
            print("Creating new plot!!!")
            if self.plot: self.plot.close()
            self.plot = aiPlot(step_value=5, calculate_avg=True)
            self.plot_count = 0

        self.plot_count += 1

        params = {
            "episode count": self.N_EPISODES.get(),
            "episode display count": self.N_EPISODES_DISPLAYED.get(),
            "learning rate": self.LR.get(),
            "gamma": self.GAMMA.get(),
            "batch size": self.BATCH_SIZE.get(),
            "epsilon start": self.EPS_START.get(),
            "epsilon decay": self.EPS_DECAY.get(),
            "save plot": self.save_plot.get(),
            "plot": self.plot,
            "plot count": self.plot_count
        }
        sim.run(params)
            

if __name__ == "__main__":
    app = DQNAdjuster()
    app.run()