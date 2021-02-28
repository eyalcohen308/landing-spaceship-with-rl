import torch
from torch import nn
from torch.distributions import Categorical
from utils import softmax


class CategoricalMLP(nn.Sequential):
    def __init__(self, n_states, n_hidden, n_actions, n_layers=2):    
        super().__init__(
            nn.Linear(n_states, n_hidden),
            nn.ReLU(),
            *sum(([nn.Linear(n_hidden, n_hidden), nn.ReLU()] for _ in range(n_layers - 2)), []),
            nn.Linear(n_hidden, n_actions)
        )
        self.args = {
            'n_states': n_states,
            'n_hidden': n_hidden,
            'n_actions': n_actions,
            'n_layers': n_layers
        }

    def policy(self, state, tau):
        q_values = self(state)

        probs = softmax(q_values.data, tau)
        #TODO: Categorical(probs).sample().item()
        # probs = probs.cpu().numpy()
        probs /= probs.sum()

        action = Categorical(probs).sample().item()
        
        return action

    @staticmethod
    def load_model(path, device='cpu'):
        checkpoint = torch.load(path)
        model = CategoricalMLP(**checkpoint['args'])
        device = torch.device(device)
        model.load_state_dict(checkpoint['model_state_dict']).to(device)
        return model, checkpoint

    def save_model(self, path, **kwargs):
        kwargs['model_state_dict'] = self.state_dict()
        kwargs['args'] = self.args
        torch.save(kwargs, path)