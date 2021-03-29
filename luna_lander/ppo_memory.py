import torch
import torch.nn as nn

import numpy as np
import random
import torch
from collections import deque


class PPOMemory:

    def __init__(self,batch_size):

        self.batch_size=batch_size

        self.memory={'state':[], 'action':[], 'log_prob':[],
                     'reward':[], 'value':[], 'not done':[]}

    def push(self, state, action, state_next, reward, not_done):

        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['log_prob'].append(state_next)
        self.memory['reward'].append(reward)
        self.memory['value'].append(state)
        self.memory['not done'].append(not_done)

    def permute_batches(self):
        self.start_batches=np.random.permutation(self.__len__()-self.batch_size)

    def generate_batch(self):

        for ind, start_batch in enumerate(self.self.start_batches):
            batch_state = torch.stack([self.memory['state'][position] for position in range(start_batch, start_batch+self.batch_size)])
            batch_action = torch.stack([self.memory['action'][position] for position in range(start_batch, start_batch+self.batch_size)])
            batch_log_prob = torch.stack([self.memory['log_prob'][position] for position in range(start_batch, start_batch+self.batch_size)])
            batch_reward = torch.stack([self.memory['reward'][position] for position in range(start_batch, start_batch+self.batch_size)])
            batch_value = torch.stack([self.memory['value'][position] for position in range(start_batch, start_batch + self.batch_size)])
            batch_not_done = torch.stack([self.memory['not done'][position] for position in range(start_batch, start_batch+self.batch_size)])
            yield batch_state, batch_action, batch_log_prob, batch_reward, batch_value, batch_not_done


    def __len__(self):

        return len(self.memory['state'])


    def reset(self):
        self.memory = {'state': [], 'action': [], 'log_prob': [],
                       'reward': [], 'value': [], 'not done': []}