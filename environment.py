#!/usr/bin/env python

import gym
from random import gauss, seed

class LunarLanderEnvironment():
    def __init__ (self, noisy=False):
        self.reward = None
        self.state = None
        self.is_terminal = None
        self.n_states = 8
        self.noisy = noisy
        self.init()

    def init(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        self.env.seed(0)
        seed(0)

    def start(self):   
        self.reward = 0.0
        self.state = self.env.reset()

        if self.noisy:
            self.state[0] += gauss(0, 0.05) # Gaussian noise
            self.state[1] += gauss(0, 0.05) # Gaussian noise
        
        self.is_terminal = False
        
        return self.state
        
    def step(self, action):

        # last_state = self.reward_obs_term[1]
        self.state, self.reward, self.is_terminal, _ = self.env.step(action)
        if self.noisy:
            self.state[0] += gauss(0, 0.05) # Gaussian noise
            self.state[1] += gauss(0, 0.05) # Gaussian noise
        # self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward, self.state, self.is_terminal