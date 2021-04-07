import torch
import torch.nn as nn
import numpy as np
import random
import torch
from collections import deque


class PPOMemory:

    def __init__(self, batch_size):
        self.batch_size = batch_size

        # list for storing transition and advantages and reward2go
        self.memory = {'state': [], 'action': [], 'log_prob': [], 'entropy': [],
                       'reward': [], 'value': [], 'not done': [], 'advantages': [], 'rewards2go': []}

    def push(self, state, action, log_prob, entropy, value, reward, not_done):
        self.memory['state'].append(state)
        self.memory['action'].append(action)
        self.memory['log_prob'].append(log_prob)
        self.memory['entropy'].append(entropy)
        self.memory['reward'].append(reward)
        self.memory['value'].append(value)
        self.memory['not done'].append(not_done)

    # generate permutation of the memory
    def permute_batches(self):
        self.start_batches = np.random.permutation(self.__len__() - self.batch_size - 1)


    def generate_batch(self):
        # generator-using yield each of batch size
        self.rand_batches = np.random.permutation(self.__len__() - self.batch_size - 1)
        for start in self.rand_batches:

            batch_state = torch.stack(self.memory['state'][start:start + self.batch_size])
            batch_action = torch.stack(self.memory['action'][start:start + self.batch_size])
            batch_log_prob = torch.stack(self.memory['log_prob'][start:start + self.batch_size])
            batch_entropy = torch.stack(self.memory['entropy'][start:start + self.batch_size])
            batch_advantages = torch.stack(self.memory['advantages'][start:start + self.batch_size])
            batch_rewards2go = torch.stack(self.memory['rewards2go'][start:start + self.batch_size])

            yield batch_state, batch_action, batch_log_prob, batch_entropy, batch_advantages, batch_rewards2go

    def __len__(self):
        return len(self.memory['state'])

    def add_advanatges_and_rewards2go(self, advantages, rewards2go):
        self.memory['advantages'] = advantages
        self.memory['rewards2go'] = rewards2go

    def reset(self):
        self.memory = {'state': [], 'action': [], 'log_prob': [], 'entropy': [],
                       'reward': [], 'value': [], 'not done': [], 'advantages': [], 'rewards2go': []}