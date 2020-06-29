from collections import namedtuple
import random


class ReplayMemory(object):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.capacity = self.batch_size*2
        self.memory = []
        self.position = 0
        self.Transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward'))

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def isFull(self):
        return len(self.memory) >= self.capacity
