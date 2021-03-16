# from threading import main_thread
from glue import RLGlue
import numpy as np
from tqdm.auto import tqdm
from agent import Agent
from environment import LunarLanderEnvironment
from objective_funcs import actor_critic, dqn
import os
from modeling import CategoricalMLP
from configs import Config
from utils import quantize_actions_list, save_rewords_plot, create_dir_if_not_exsits


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
        
        create_dir_if_not_exsits(save_path)
        
        max_reward = - float('inf')
        max_path = ''

        for episode in tqdm(range(start_episode, args.goal_episodes)):

            reward = glue.episode(args.timeout)
            
            rewards[run, episode - start_episode] = reward
            
            if reward > max_reward:
                os.remove(max_path) if os.path.exists(max_path) else None

                max_path = os.path.join(save_path, f'best_model_r{run}_e{episode+1}_reward{int(reward)}.pth')
                glue.agent.model.save_model(max_path, episode=episode)
                max_reward = reward

            if episode == args.goal_episodes - 1:
                path = os.path.join(save_path, f'last_model_e{episode+1}_run{run}_reward{reward}.pth')
                glue.agent.model.save_model(path, episode=episode)
                
            print('Run:{}, episode:{}, reward:{}'.format(run, episode, reward))

    save_rewords_plot(save_path, rewards)

    print("Finished training, Existing...ֿ\n")
    return rewards


def main(args):

    environment = LunarLanderEnvironment(noisy=args.noisy)
    objectives = {
        'dqn': dqn,
        'ac2': actor_critic,
        'ppo': None
    }

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
        'device': args.device,
        'batch_size': 100,
        'buffer_size': 50000,
        'gamma': 0.99,
        'lr': args.lr,
        'tau':0.01 ,
        'seed':0,
        'num_replay_updates':5,
        'objective_func': objectives.get(args.objective),
        'actions': actions,
        'priority': args.priority,
        'no_experience': args.no_experience
    }

    agent = Agent(agent_configs)
    sum_reward = train(environment, agent, args)

if __name__ == "__main__":

    args = Config().parse()
    main(args)