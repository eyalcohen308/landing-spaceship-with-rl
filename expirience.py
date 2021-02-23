from torch import tensor, stack
import random

class ExperienceBuffer:
    
    def __init__(self, batch_size, buffer_size, seed=None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size     
        self.buffer = []
        if seed:
            random.seed(seed)
        
    def append(self, state, action, terminal, reward, next_state):
        if len(self.buffer) == self.buffer_size:
            del self.buffer[0]
        self.buffer.append((tensor(state), tensor(action), tensor(terminal), tensor(reward).float(), tensor(next_state)))
        

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        batch = tuple(map(stack, zip(*batch)))
        return batch


    def get_buffer(self):
        return self.buffer
    
    def is_full(self):
        return len(self.buffer) >= self.batch_size
        