import torch
import torch.nn as nn
import torch.nn.functional as F
import os

"""
The Model Class:
    This Model is built on top of torch's nn.Module, which is a base class used for building
    most basic neural networks. The following model is feed forward, meaning it will only
    process the input in one direction (input -> hidden layers -> output).
"""
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
    """
    save() - Saves the model's state dictionary to the directory "./models"

    Args:
        file_name: The name to save the dictionary as.
    """
    def save(self, file_name='model.pth'):
        model_dir_path = "./models"
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        
        file_name = os.path.join(model_dir_path, file_name)
        torch.save(self.state_dict(), file_name)