from torch import nn

# class OLD_MLP(nn.Module):
    
#     def __init__(self, network_arch):
        
#         super().__init__()
#         self.num_states = network_arch['num_states']
#         self.hidden_units = network_arch['num_hidden_units']
#         self.num_actions = network_arch['num_actions']
        
#         # The hidden layer
#         self.fc1 = nn.Linear(in_features = self.num_states, out_features = self.hidden_units)
        
#         # The output layer
#         self.fc2 = nn.Linear(in_features = self.hidden_units, out_features = self.num_actions)
        
#     def forward(self, x):
        
#         x = F.relu(self.fc1(x))
        
#         # No activation func, output should be a tensor(batch, num_actions)
#         out = self.fc2(x)
        
#         return out

class MLP(nn.Sequential):
    def __init__(self, n_states, n_hidden, n_actions):    
        super().__init__(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_actions)
        )