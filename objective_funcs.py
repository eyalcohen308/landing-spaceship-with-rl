import torch
from utils import softmax
from torch import nn
from torch.autograd import Variable

mse = nn.MSELoss()
def actor_critic(batch, model, current_model, discount, tau):
    
    states, actions, terminals, rewards, next_states = batch
    
    
#     print(next_states)
    q_next = current_model(Variable(next_states)).squeeze()
    probs = softmax(q_next, tau)

    # calculate the maximum action value of next states
#     expected_q_next = (1-torch.stack(terminals)) * (torch.sum(probs * q_next , axis = 1))
    max_q_next = (1-terminals) * (torch.max(q_next , axis = 1)[0])
    # calculate the targets
    
    rewards = rewards.float()
#     targets = Variable(rewards + (discount * expected_q_next)).float()
    targets = Variable(rewards + (discount * max_q_next)).float()
    
    # calculate the outputs from the previous states (batch_size, num_actions)
    outputs = model(Variable(states.float())).squeeze()
    

    actions = actions.view(-1,1)
    
    outputs = torch.gather(outputs, 1, actions).squeeze()
    # the loss
    loss = mse(outputs, targets)
    return loss