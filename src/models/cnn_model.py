import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, config):
        super(CNNModel, self).__init__()
        self.config = config
        
        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate CNN output dimension
        self.cnn_output_dim = self._get_conv_output_dim((96, 96, 3))
        
        # Actor (Policy) head
        self.actor = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3 actions: steering, gas, brake
        )
        
        # Critic (Value) head
        self.critic = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def _get_conv_output_dim(self, shape):
        input_dummy = torch.zeros(1, *shape).permute(0, 3, 1, 2)
        x = F.relu(self.conv1(input_dummy))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return int(np.prod(x.shape[1:]))
    
    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        
        # Prepare input
        x = state.permute(0, 3, 1, 2) / 255.0  # Normalize and reshape for CNN
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, self.cnn_output_dim)
        
        # Get action distributions and value
        action_mean = torch.tanh(self.actor(x))  # Use tanh to bound actions
        value = self.critic(x)
        
        return action_mean, value
    
    def act(self, state, deterministic=False):
        with torch.no_grad():
            action_mean, value = self.forward(state)
            if deterministic:
                action = action_mean
            else:
                # Add some exploration noise
                action = action_mean + torch.randn_like(action_mean) * 0.1
                action = torch.clamp(action, -1, 1)
        
        return action, value

    def evaluate_actions(self, state, actions):
        action_mean, value = self.forward(state)
        return value, action_mean