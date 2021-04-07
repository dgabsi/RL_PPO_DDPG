import numpy as np
import random
import torch
from collections import deque


class ReplayMemory:

    def __init__(self, capacity):

        self.capacity = capacity

        self.memory={'state':deque(maxlen=self.capacity), 'action':deque(maxlen=self.capacity), 'state_next':deque(maxlen=self.capacity),
                     'reward':deque(maxlen=self.capacity), 'not done':deque(maxlen=self.capacity)}

    def push(self, state, action, state_next, reward, not_done):

        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['state_next'].append(state_next)
        self.memory['reward'].append(reward)
        self.memory['not done'].append(not_done)

    def sample_batch(self,batch_size):


        curr_length=min(self.__len__(), self.capacity)

        batch_positions=random.sample(list(np.arange(curr_length)), batch_size)

        batch_state=torch.stack([self.memory['state'][position] for position in batch_positions])
        batch_action = torch.stack([self.memory['action'][position] for position in batch_positions])
        batch_state_next = torch.stack([self.memory['state_next'][position] for position in batch_positions])
        batch_reward = torch.stack([self.memory['reward'][position] for position in batch_positions])
        batch_not_done = torch.stack([self.memory['not done'][position] for position in batch_positions])


        return batch_state,batch_action, batch_state_next, batch_reward,batch_not_done

    def __len__(self):

        return len(self.memory['state'])



