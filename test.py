import os

# Gym packages
import gym
import matplotlib.pyplot as plt
# Pytorch packages
import torch
from matplotlib.offsetbox import AnchoredText
from torch.autograd import Variable

from agent import Agent
from configs import Config
# Model import
from modeling import CategoricalMLP
# other packages:
from utils import create_dir_if_not_exsits, quantize_actions_list
import pandas as pd

def save_test_plot(rewords, plot_path, model_name):
    rewords = [round(elem, 2) for elem in rewords]
    s = pd.Series(rewords)
    ax = s.plot.kde()
    anchored_text = AnchoredText(f"Avg: {round(s.mean(), 2)}\nMed: {round(s.median(), 2)}", loc=2)
    ax.add_artist(anchored_text)
    ax.set_xlabel('Values')
    plt.savefig(plot_path)

def test(config):
    model_path = config.model_path
    test_n_runs = config.test_n_runs


    test_plot_dir = os.path.join(config.output_dir,config.name)
    test_plot_path = os.path.join(test_plot_dir, config.test_plot_name)
    
    create_dir_if_not_exsits(test_plot_dir)

    actions = quantize_actions_list(config.run_min_states)

    # model_path = '/content/landing-spaceship-with-rl/outputs/dqn_lr5_h1024/last_model_e1000_run2_reward288.7133085145376.pth'

    model, _ = CategoricalMLP.load_model(model_path)

    agent_configs = {
            'model': model,
            'device': config.device,
            'batch_size': None,
            'buffer_size': None,
            'gamma': None,
            'lr': config.lr,
            'tau':0.01 ,
            'seed':0,
            'num_replay_updates':None,
            'objective_func': None,
            'actions': actions,
            'priority': args.priority
        }

    agent = Agent(agent_configs)
    env = gym.make("LunarLanderContinuous-v2")

    total_rewords = []
    for i_episode in range(test_n_runs):
        observation = env.reset()
        total_reward = 0
        
        for t in range(1000):
            # env.render()
            with torch.no_grad():
                observation = Variable(torch.tensor(observation).view(1, -1))
                action = agent.act(observation)
                observation, reward, done, info = env.step(action)
                total_reward += reward

                if done:
                    print(f"Episode {i_episode} finished after {t+1} timesteps, total reward : {total_reward}")
                    total_rewords.append(total_reward)
                    break
    env.close()

    print('Saved rewords plot.')
    save_test_plot(total_rewords, test_plot_path, config.name)
    print('Test Ended, Existing...')

def main(args):
    test(args)


if __name__ == "__main__":

    args = Config().parse()
    main(args)
