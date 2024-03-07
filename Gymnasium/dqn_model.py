import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
Linear networks use only linear transformations in the layers,
and has an input layer, some hidden layer(s), and an output layer.

The size of the input layer is represented by the size of the observation
space,and the size of the output layer is represented by the size of the 
action space.

This class is a subclass of torch.nn.Module, which is a base class used
"""
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    """
    forward() - used to get predictions/actions from the model by running the input "x"
    through all layers of the NN. A rectified linear unit (RELU) function is used since
    the model is linear, and RELU intruduces non-linearity which can help the model learn
    more complex patterns.
    """
    def forward(self, x):
        # Run input tensor through each layer, introducing a RELU function mid-way.
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

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

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        # Use torch's Adam optimizer. The optimizer is responsible for
        # improving the performance of the model by tweaking the parameters.
        self.optim = optim.Adam(model.parameters(), lr=self.lr)

        # Use the mean squared error function for determining loss (Q_new - Q_old)^2
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        # Convert all inputs except "done" to tensor
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Predict Q value with current state
        prediction = self.model(state)

        target = prediction.clone()
        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))

            target[i][torch.argmax(action).item()] = Q_new

        # Q_new = reward + gamma * max(prediction)
            
        self.optim.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optim.step()
        

