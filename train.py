from threading import main_thread
from glue import RLGlue
import numpy as np
from tqdm.auto import tqdm
import torch
from agent import Agent
from environment import LunarLanderEnvironment
from objective_funcs import actor_critic
import os

def run_experiment(environment, agent, experiment_configs, finetune, checkpoint_path='new_results/current_model_700.pth', save_path='new_results2'):
    
    glue = RLGlue(environment, agent)
    
    agent_sum_reward = np.zeros((experiment_configs['num_runs'],experiment_configs['num_episodes']))
    
    for run in tqdm(range(experiment_configs['num_runs'])):

        glue.init(run)
        
        if finetune:
            
            checkpoint = torch.load(checkpoint_path)
            glue.agent.model.load_state_dict(checkpoint['model_state_dict'])

            start_episode = checkpoint['episode'] + 1
            
            print(f'Finetuning {checkpoint_path}')
        else:
            start_episode = 0
            print('Training from scratch...')
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for episode in tqdm(range(start_episode, start_episode + experiment_configs['num_episodes'])):

            reward = glue.episode(experiment_configs['timeout'])
            
            agent_sum_reward[run, episode - start_episode] = reward
            
            if episode == start_episode + experiment_configs['num_episodes'] - 1:
                
                current_model = glue.agent.model
                path = os.path.join(save_path, 'current_model_{}.pth'.format(episode+1))
                torch.save({'episode':episode,
                'model_state_dict':current_model.state_dict(),
                            },
                path)
                
            print('Run:{}, episode:{}, reward:{}'.format(run, episode, reward))
            
    return agent_sum_reward

def main():
    agent_configs = {
        'model_args' : {'n_states':8,
                'n_hidden' : 256,
                'n_actions': 4},
        
        'batch_size': 100,
        'buffer_size': 50000,
        'gamma': 0.99,
        'lr': 1e-4,
        'tau':0.01 ,
        'seed':0,
        'num_replay_updates':5,
        'objective_func': actor_critic
        
    }

    experiment_configs = {
        'num_runs':1,
        'num_episodes':100,
        'timeout': 1000
    }
    device = 'cuda:0'
    agent = Agent(agent_configs, device)
    environment = LunarLanderEnvironment()

    PATH = 'new_results/current_model_700.pth'
    sum_reward = run_experiment(environment, agent, experiment_configs, finetune=False)

if __name__ == "__main__":
    main()