import torch
import torch.nn as nn
import torch.optim as optim
from traitlets import Int
# import torch.nn.function as F


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        self.obs_shape = obs_shape # env.observation_space.shape (4,) 
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256), # fully connected layer
            nn.ReLU(), # activation layer
            nn.Linear(256, num_actions), # final layer 
            # we dont want activation in last because we are approximating the real q values
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)

    def forward(self, x):
        return self.net(x) 
        # x pode (N, obs_shape) output será (N, num_actions)

