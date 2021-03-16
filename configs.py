import argparse

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # Train
        self.parser.add_argument('--name', action='store',default='models', type=str, help='run name.')
        self.parser.add_argument('--finetune', default=False, action='store_true', help='use finetune if specified.')
        self.parser.add_argument('--run_min_states', default=False, action='store_true', help='run on min states(8 states).')
        self.parser.add_argument('--output_dir', action='store', default='outputs',type=str , help='output directory for saved model checkpoint.')
        self.parser.add_argument('--n_runs',type=int, action='store', default=1, help='# program runs.')
        self.parser.add_argument('--n_layers',type=int, action='store', default=3, help='# of layers runs.')
        self.parser.add_argument('--n_hidden',type=int, action='store', default=1024, help='# of hidden in the net.')
        self.parser.add_argument('--goal_episodes',type=int, default=1000, help='number of episodes to run.')
        self.parser.add_argument('--timeout',type=int, default=1000, help='timout value.')
        self.parser.add_argument('--lr',type=float, default=1e-5, help='lr value.')
        self.parser.add_argument('--checkpoint', default=None, help='checkpoint path for finetuning.')
        self.parser.add_argument('--device', type=str, default='cpu', help='timout value.')
        self.parser.add_argument('--objective', type=str, default='dqn', choices=['dqn', 'ac2', 'ppo'], help='select objective function to work with.')
        self.parser.add_argument('--noisy', default=False, action='store_true', help='add gaussian noise to spaceship location observation.')
        self.parser.add_argument('--priority', default=False, action='store_true', help='use priority learning mechanism.')
        self.parser.add_argument('--no_experience', default=False, action='store_true', help='use priority learning mechanism.')

        # Test:
        self.parser.add_argument('--model_path', action='store',default=None, type=str, help='saved model path.')
        self.parser.add_argument('--test_n_runs',type=int, action='store', default=100, help='# of tests trails.')
        self.parser.add_argument('--test_plot_name', action='store',default='test_rewords_plot.jpg', type=str, help='test rewords plot name.')
    
    def parse(self):
        self.conf = self.parser.parse_args()
        return self.conf