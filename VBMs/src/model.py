import torch
import torch.nn as nn
import torch.optim as optim
from traitlets import Int
# import torch.nn.function as F


class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=1e-4):
        super(Model, self).__init__()
        self.obs_shape = obs_shape # env.observation_space.shape (4,) 
        self.num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_shape[0], 256), # fully connected layer
            nn.ReLU(), # activation layer
            nn.Linear(256, num_actions), # final layer 
            # we dont want activation in last because we are approximating the real q values
        )
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x) 
        # x pode (N, obs_shape) output será (N, num_actions)

class ConvModel(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=1e-4, dueling=True):
        super(ConvModel, self).__init__()
        self.dueling = dueling
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        # self.conv_net = nn.Sequential(
        #     nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(8, 8), stride=(4, 4)),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2)),
        #     nn.ReLU(),
        # )

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

        if not dueling:
            self.fc_net = nn.Sequential(
                nn.Linear(fc_input_size, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            # changed from 256 to 512
            self.fc_net = nn.Sequential(
                nn.Linear(fc_input_size, 512),
                nn.ReLU(),
            )

            self.v_net = nn.Sequential( # V(s)
                nn.Linear(512, 1),
            )

            self.action_advantage_net = nn.Sequential( # A(s, a)
                nn.Linear(512, num_actions),
            )



        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, x):
        x = x.float()/255.0
        conv_latent = self.conv_net(x) # mudar shape (N, 4, 84, 84) para (N, 4*84*84)
        # return self.fc_net(conv_latent.view((conv_latent.shape[0], -1)))
        hidden_layer_out = self.fc_net(conv_latent.flatten(1))
        if not self.dueling:
            return self.fc_net(conv_latent.flatten(1))
        else:
            v = self.v_net(hidden_layer_out)
            actions_advantage = self.action_advantage_net(hidden_layer_out)

            q = v + actions_advantage - torch.mean(actions_advantage, 1, keepdim=True) # Q(s,a)
            
            return q
        # try:
        #     # conv_latent.shape ==torch.Size([300, 32, 9, 9])
        #     return self.fc_net(conv_latent.flatten(1))
        # except:
        #     # conv_latent.shape ==torch.Size([32, 9, 9])
        #     return self.fc_net(conv_latent.flatten(0))
    
