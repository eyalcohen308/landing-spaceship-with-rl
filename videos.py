# Basic packages
import os

from tqdm.auto import tqdm

import numpy as np

import matplotlib.pyplot as plt

# Model import
from modeling import CategoricalMLP

# Pytorch packages
import torch
import torch.nn as nn
from torch.autograd import Variable

# Gym packages
import gym
from gym import wrappers
from time import time

# Display packages:
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1400, 900))
display.start()

# This code creates a virtual display to draw game images on. 
# If you are running locally, just ignore it

# other packages:
from environment import LunarLanderEnvironment

from utils import softmax
from utils import quantize_actions_list
from objective_funcs import dqn
from agent import Agent


def create_video(model_path):

    actions = quantize_actions_list()

    # model_path = '/content/landing-spaceship-with-rl/outputs/dqn_lr5_h1024/last_model_e1000_run2_reward288.7133085145376.pth'

    model, _ = CategoricalMLP.load_model(model_path)

    agent_configs = {
                'model': model,
                'device': 'cpu',
                'batch_size': None,
                'buffer_size': None,
                'gamma': None,
                'lr': 1e-5,
                'tau':0.01 ,
                'seed':0,
                'num_replay_updates':None,
                'objective_func': None,
                'actions': actions,
                'priority': False,
                'no_experience': True
            }

    agent = Agent(agent_configs)
    env = gym.make("LunarLanderContinuous-v2")
    env = wrappers.Monitor(env, './videos/' + '/', force=True)

    for i_episode in range(10):
        observation = env.reset()
        total_reward = 0
        
        for t in range(1000):
            env.render()
    #         print(observation)
            with torch.no_grad():
                observation = Variable(torch.tensor(observation).view(1, -1))
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    
                    print("Episode finished after {} timesteps, total reward : {}".format(t+1, total_reward))
                    break
    env.close()
    print('end')

def play_video(video_path):
    from IPython.display import HTML
    from base64 import b64encode
    mp4 = open(video_path,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
    <video width=400 controls>
        <source src="%s" type="video/mp4">
    </video>
    """ % data_url)