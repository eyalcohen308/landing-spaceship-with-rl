
from modeling import CategoricalMLP
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

    def __init__(self, agent_config):
        #TODO: change init_agent to __init__ every place we use init_agent.
        self.config = agent_config
        self.init(self.config['seed'])
    
    def init(self, seed):
        self.device = torch.device(self.config['device'])
        self.model = self.config['model'].to(self.device)
        self.actions = self.config['actions']
        self.buffer = ExperienceBuffer(self.config['batch_size'],
                            self.config['buffer_size'],
                            seed,
                            priority=self.config['priority'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr = self.config['lr'], 
                                          betas = [0.99,0.999], 
                                          eps = 1e-04)
    
        self.batch_size = self.config['batch_size']
        self.discount = self.config['gamma']
        self.tau = self.config['tau']
        self.num_replay = self.config['num_replay_updates']
        self.n_actions = len(self.config['actions'])
        self.no_experience= self.config['no_experience']

        assert self.n_actions == len(self.actions)
        
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
        self.last_action = action
        
        return self.actions[action]
    
    def policy(self, state):
        return self.model.policy(state, self.tau)

    def act(self, state):
        action = self.policy(state)
        return self.actions[action]
    
    def replay_learn(self):
        current_model = deepcopy(self.model)
        
        for i in range(self.num_replay):
            batch, indices, weights = self.buffer.sample()
            batch = [ten.to(self.device) for ten in batch]
            self.optimizer.zero_grad()
            loss = self.objective_func(batch, self.model, current_model, self.discount, self.tau)

            if self.buffer.priority:
                self.buffer.priorities[indices] = loss.sqrt().cpu().detach().numpy()
                loss = loss * torch.tensor(weights).to(self.device)

            loss.mean().backward()
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
        self.last_action = action
        
        return self.actions[action]
    
    def end(self, reward):
        if self.no_experience:
            self.episode_steps = self.episode_steps + 1
        else:
            self.episode_steps += 1
        self.sum_rewards += reward

        state = torch.zeros_like(self.last_state)
        is_terminal = 1
        self.buffer.append(self.last_state, self.last_action, is_terminal, reward, state)
        
        if self.buffer.is_full():
            self.replay_learn()

        ### Save the model at each episode
        
        # torch.save(self.model, 'new_results/current_nodel.pth')
    def load_model(self, check_point):
        self.model, checkpoint = self.model.load_model(check_point)
        self.model = self.model.to(self.device)
        return checkpoint

    def agent_message(self, message):
        if message == 'get_sum_reward':
            
            return self.sum_rewards
        else:
            raise Exception('No given message of the agent!')
