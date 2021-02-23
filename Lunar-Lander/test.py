# Basic packages
import os

import shutil

from collections import deque

from copy import deepcopy

from tqdm.auto import tqdm

import numpy as np

import matplotlib.pyplot as plt

# Lunar lander and RL-GLUE packages
from rl_glue import RLGlue
from environment import BaseEnvironment

from lunar_lander import LunarLanderEnvironment

from agent import BaseAgent

from plot_script import plot_result, draw_neural_net

# Pytorch packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Gym packages
import gym
from gym import wrappers
from time import time




model_path = 'new_results2/current_model_800.pth'
current_model = RLModel(agent_configs['network_arch'])
checkpoint = torch.load(model_path)
current_model.load_state_dict(checkpoint['model_state_dict'])

def policy(state, model, num_actions = 4):
        
        """
        Select the action given a single state.
        
        """
        model.eval()
        # compute action values states:(1, state_dim)
        q_values = model(state)

        # compute the probs of each action (1, num_actions)
        probs = softmax(q_values.data, tau = 0.01)
        probs = np.array(probs)
        probs /= probs.sum()

        # select action
        rand_generator = np.random.RandomState(seed = 1)
        action = rand_generator.choice(num_actions, 1, p = probs.squeeze())
#         action = np.argmax(probs.squeeze())
        
        return action

env = gym.make("LunarLander-v2")
env = wrappers.Monitor(env, './videos_800/' + '/')
for i_episode in range(10):
    observation = env.reset()
    total_reward = 0
    
    for t in range(1000):
        env.render()
#         print(observation)
        with torch.no_grad():
            observation = Variable(torch.tensor(observation).view(1, -1))
            action = policy(observation, current_model)
            action = int(action.squeeze())

            observation, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                
                print("Episode finished after {} timesteps, total reward : {}".format(t+1, total_reward))
                break
env.close()
print('end')