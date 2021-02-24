#!/usr/bin/env python

import numpy as np
import gym

class LunarLanderEnvironment():
    def __init__ (self):
        self.reward = None
        self.state = None
        self.is_terminal = None
        self.n_states = 8
        self.init()

    def init(self):
        self.env = gym.make("LunarLanderContinuous-v2")
        self.env.seed(0)

    def start(self):   
        self.reward = 0.0
        self.state = self.env.reset()
        self.is_terminal = False
        
        return self.state
        
    def step(self, action):

        # last_state = self.reward_obs_term[1]
        self.state, self.reward, self.is_terminal, _ = self.env.step(action)
        
        # self.reward_obs_term = (reward, current_state, is_terminal)
        
        return self.reward, self.state, self.is_terminal