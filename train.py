# from threading import main_thread
from glue import RLGlue
import numpy as np
from tqdm.auto import tqdm
import torch
from agent import Agent
from environment import LunarLanderEnvironment
from objective_funcs import dqn
import os
from modeling import CategoricalMLP
import argparse
import sys
import itertools
from utils import quantize_actions_list
import pandas as pd


def train(environment, agent, args):
    save_path = os.path.join(args.output_dir,args.name)
    glue = RLGlue(environment, agent)
    
    rewards = np.zeros((args.n_runs, args.goal_episodes))
    
    for run in tqdm(range(args.n_runs)):

        glue.init(run)
        
        if args.checkpoint:
            checkpoint = glue.agent.load_model(args.checkpoint)

            start_episode = checkpoint['episode'] + 1
            
            print(f'Finetuning {args.checkpoint}')
        else:
            start_episode = 0
            print('Training from scratch...')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        max_reward = - float('inf')
        max_path = ''

        for episode in tqdm(range(start_episode, args.goal_episodes)):

            reward = glue.episode(args.timeout)
            
            rewards[run, episode - start_episode] = reward
            
            if reward > max_reward:
                os.remove(max_path) if os.path.exists(max_path) else None

                max_path = os.path.join(save_path, f'best_model_run{run}_reward{reward}.pth')
                glue.agent.model.save_model(max_path, episode=episode)
                max_reward = reward

            if episode == args.goal_episodes - 1:
                path = os.path.join(save_path, f'last_model_e{episode+1}_run{run}_reward{reward}.pth')
                glue.agent.model.save_model(path, episode=episode)
                
            print('Run:{}, episode:{}, reward:{}'.format(run, episode, reward))

    path = os.path.join(save_path, f'plot_rewards.jpg')
    pd.DataFrame(rewards).T.plot.line().figure.savefig(path)

    print("Finished training, Existing...ֿ\n")
    return rewards


def main(args):

    environment = LunarLanderEnvironment()

    actions = quantize_actions_list(args.run_min_states)
    model_args = {
                'n_states':environment.n_states,
                'n_hidden' : args.n_hidden,
                'n_actions': len(actions),
                'n_layers': args.n_layers
                }

    model = CategoricalMLP(**model_args)

    agent_configs = {
        'model': model,
        'model_args' : model_args,
        'device': args.device,
        'batch_size': 100,
        'buffer_size': 50000,
        'gamma': 0.99,
        'lr': args.lr,
        'tau':0.01 ,
        'seed':0,
        'num_replay_updates':5,
        'objective_func': dqn,
        'actions': actions
    }

    agent = Agent(agent_configs)
    sum_reward = train(environment, agent, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', action='store',default='models', type=str, help='run name.')
    parser.add_argument('--finetune', default=False, action='store_true', help='use finetune if specified.')
    parser.add_argument('--run_min_states', default=False, action='store_true', help='use finetune if specified.')
    parser.add_argument('--output_dir', action='store', default='outputs',type=str , help='output directory for saved model checkpoint.')
    parser.add_argument('--n_runs',type=int, action='store', default=1, help='# program runs.')
    parser.add_argument('--n_layers',type=int, action='store', default=256, help='# of layers runs.')
    parser.add_argument('--n_hidden',type=int, action='store', default=2, help='# of hidden in the net.')
    parser.add_argument('--goal_episodes',type=int, default=700, help='number of episodes to run.')
    parser.add_argument('--timeout',type=int, default=1000, help='timout value.')
    parser.add_argument('--lr',type=float, default=1e-5, help='lr value.')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path for finetuning.')
    parser.add_argument('--device', type=str, default='cpu', help='timout value.')

    args = parser.parse_args()
    main(args)