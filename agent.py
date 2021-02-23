
from modeling import MLP
import torch
from torch import nn
from expirience import ExperienceBuffer
import numpy as np
from utils import softmax
from copy import deepcopy

"""
agent_configs = {
    'model_args' : {'n_states':8,
               'n_hidden' : 256,
               'n_actions': 4},
    
    'batch_size': 8,
    'buffer_size': 50000,
    'gamma': 0.99,
    'lr': 1e-4,
    'tau':0.01 ,
    'seed':0,
    'num_replay_updates':5
    'objective_func': actor_critic
      
}
"""

class Agent():

    def __init__(self, agent_config, device='cpu'):
        #TODO: change init_agent to __init__ every place we use init_agent.
        self.config = agent_config
        self.device = torch.device(device)
        self.init(self.config['seed'])

    def policy(self, state):
        q_values = self.model(state)

        probs = softmax(q_values.data, self.tau)
        probs = probs.cpu().numpy()
        probs /= probs.sum()

        action = self.rand_generator.choice(self.n_actions, 1, p = probs.squeeze())
        
        return action
    
    def init(self, seed):
        self.model = MLP(**self.config['model_args']).to(self.device)
        self.buffer = ExperienceBuffer(self.config['batch_size'],
                            self.config['buffer_size'],
                            seed)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr = self.config['lr'], 
                                          betas = [0.99,0.999], 
                                          eps = 1e-04)
    
        self.batch_size = self.config['batch_size']
        self.discount = self.config['gamma']
        self.tau = self.config['tau']
        self.num_replay = self.config['num_replay_updates']
        self.n_actions = self.config['model_args']['n_actions']

        self.rand_generator = np.random.RandomState(seed)
        self.objective_func = self.config['objective_func']
        
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0
       
    def start(self, state):
        #TODO: change agent_start to start
        self.sum_rewards = 0
        self.episode_steps = 0
        
        state = torch.tensor([state], device=self.device).view(1, -1)

        action = self.policy(state)
        
        self.last_state = state
        self.last_action = int(action)
        
        return self.last_action
    
    # def sample_batch(self):
    #     batch = self.buffer.sample()
        
    #     # batch = [torch.tensor(instance, device=self.device) for instance in zip(*batch)]
    #     # batch = [torch.tensor(instance, device=self.device) for instance in map(list, zip(*batch))]
    #     batch = [instance for instance in map(list, zip(*batch))]
    #     return batch

    def replay_learn(self):
        current_model = deepcopy(self.model)
        
        for i in range(self.num_replay):
            batch = self.buffer.sample()
            batch = [ten.to(self.device) for ten in batch]
            self.optimizer.zero_grad()
            loss = self.objective_func(batch, self.model, current_model, self.discount, self.tau)
            loss.backward()
            self.optimizer.step()

    def step(self, reward, state):
       #TODO: change agent_start to start
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.tensor([state], device=self.device)
        
        action = self.policy(state)
        is_terminal = 0
        self.buffer.append(self.last_state, self.last_action, is_terminal, reward, state)
        
        if self.buffer.is_full():
            self.replay_learn()
                
        self.last_state = state
        self.last_action = int(action)
        
        return self.last_action
    
    def end(self, reward):
        #TODO: change agent_start to start
        self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.zeros_like(self.last_state)
        is_terminal = 1
        self.buffer.append(self.last_state, self.last_action, is_terminal, reward, state)
        
        if self.buffer.is_full():
            self.replay_learn()

        ### Save the model at each episode
        
        # torch.save(self.model, 'new_results/current_nodel.pth')

    def agent_message(self, message):
        if message == 'get_sum_reward':
            
            return self.sum_rewards
        else:
            raise Exception('No given message of the agent!')
