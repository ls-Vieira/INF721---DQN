import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim

class DeepQNetwork(nn.Module):
  def __init__(self, learning_rate, state_dim, n_actions):
    super(DeepQNetwork, self).__init__()
    
    self.state_dim = state_dim
    self.n_actions = n_actions

    self.fc1 = nn.Linear(self.state_dim, 24)
    self.fc2 = nn.Linear(24, 24)
    self.fc3 = nn.Linear(24, 12)
    self.fc4 = nn.Linear(12, 12)
    self.fc5 = nn.Linear(12, self.n_actions)
    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.loss = nn.HuberLoss()

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


  def forward(self, state):
    x = torch.relu(self.fc1(state))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    x = torch.relu(self.fc4(x))
    actions = self.fc5(x)

    return actions