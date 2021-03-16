from numpy.random import beta
import torch
from torch import tensor, stack
import random
import numpy as np

class ExperienceBuffer:
    
    def __init__(self, batch_size, buffer_size, seed=None, priority=False, alpha=0.5, beta =0.5):
        self.alpha = alpha
        self.buffer_size = buffer_size
        self.batch_size = batch_size     
        self.buffer = []
        self.priority = priority
        if priority:
            self.priorities = np.empty(0)
            self.weights = np.empty(0)
        self.beta = beta
        if seed:
            random.seed(seed)
        
    def append(self, state, action, terminal, reward, next_state):
        if len(self.buffer) == self.buffer_size:
            del self.buffer[0]
            if self.priority:
                self.priorities = np.delete(self.priorities, 0)
                self.weights = np.delete(self.weights, 0)

        self.buffer.append((tensor(state), tensor(action), tensor(terminal), tensor(reward).float(), tensor(next_state)))

        if self.priority:
            self.priorities = np.append(self.priorities,self.priorities.max() if self.priorities.size else 1)
            self.weights = np.append(self.weights, 1)
        

    def sample(self):
        # batch = random.sample(self.buffer, self.batch_size)
        batch_indices = None
        weights = None

        if not self.priority:
            batch_values = random.sample(self.buffer, self.batch_size)

        else:
            p = self.priorities ** self.alpha
            p = p/p.sum()
            # batch = random.choices(self.buffer, weights=weights, k=self.batch_size)
            batch_indices = np.random.choice(len(self.buffer), self.batch_size, p=p)
            batch_values = [self.buffer[indice] for indice in batch_indices]

            weights = (len(self.buffer) * p[batch_indices] ** (-self.beta)) / self.weights.max()
            self.weights[batch_indices] = weights

        batch = tuple(map(stack, zip(*batch_values)))
        return batch, batch_indices, weights


    def get_buffer(self):
        return self.buffer
    
    def is_full(self):
        return len(self.buffer) >= self.batch_size

    def update_priorities(self, tds, indices):
        N = len(self.buffer)
        for td, index in zip(tds, indices):
            # N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority**self.alpha - old_priority**self.alpha
            updated_probability = td[0]**self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index) 
            self.memory_data[index] = data