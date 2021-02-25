import torch
import os
import itertools
import numpy as np


def softmax(action_values, tau=1.0):
    max_action_value = torch.max(action_values, axis = 1, keepdim = True)[0]/tau
    action_values = action_values/tau
    
    preference = action_values - max_action_value
    
    exp_action = torch.exp(preference)
    sum_exp_action = torch.sum(exp_action, axis = 1).view(-1,1)


    probs = exp_action/sum_exp_action

    return probs


def quantize_actions_list(minimize=False):
    if not minimize:
        main_engine = [-1,0.25,0.5, 0.75,1]
        right_and_left_engine = [0, -1, -0.75, 0.75, 1]

    else:
        main_engine = [-1,1]
        right_and_left_engine = [-1,-0.75,0,0.75,1]
    
    actions = itertools.product(main_engine, right_and_left_engine)
    actions = [np.array(action, dtype=np.float32) for action in actions]
    return actions
