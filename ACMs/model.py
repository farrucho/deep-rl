import torch
import torch.nn as nn
import torch.optim as optim
from traitlets import Int
# import torch.nn.function as F



class PolicyReinforceModel(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=1e-4):
        super(PolicyReinforceModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        # we need to calculate the input of the fc_net, that is the output of the conv net
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            x = self.conv_net(dummy)
            fc_input_size = x.shape[1]*x.shape[2]*x.shape[3] # the first dimension is batch size, input: (N,Cin,Hin,Win) Output: (N,Cout,Hout,Wout)

        self.fc_net = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = x.float()/255.0
        conv_latent = self.conv_net(x) # mudar shape (N, 4, 84, 84) para (N, 4*84*84)

        return self.fc_net(conv_latent.flatten(1))
    
class PolicySimpleReinforceModel(nn.Module):
    # used for cartpole
    def __init__(self, obs_shape, num_actions, lr=1e-4):
        super(PolicySimpleReinforceModel, self).__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256), # fully connected layer
            nn.ReLU(), # activation layer
            nn.Linear(256, num_actions), # final layer
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
    
    def forward(self, x):
        return self.net(x)

class ValueSimpleStateModel(nn.Module):
    # used for cartpole
    def __init__(self, obs_shape, lr=1e-4):
        super(ValueSimpleStateModel, self).__init__()
        self.obs_shape = obs_shape

        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256), # fully connected layer
            nn.ReLU(), # activation layer
            nn.Linear(256, 1), # final layer
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
    
    def forward(self, x):
        return self.net(x)

class ValueStateModel(nn.Module):
    def __init__(self, obs_shape, lr=1e-4):
        super(ValueStateModel, self).__init__()
        self.obs_shape = obs_shape

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(),
        )

        # we need to calculate the input of the fc_net, that is the output of the conv net
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            x = self.conv_net(dummy)
            fc_input_size = x.shape[1]*x.shape[2]*x.shape[3] # the first dimension is batch size, input: (N,Cin,Hin,Win) Output: (N,Cout,Hout,Wout)

        self.fc_net = nn.Sequential(
            nn.Linear(fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = x.float()/255.0
        conv_latent = self.conv_net(x) # mudar shape (N, 4, 84, 84) para (N, 4*84*84)

        return self.fc_net(conv_latent.flatten(1))