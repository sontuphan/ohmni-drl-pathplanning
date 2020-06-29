import math
import random
import torch


class Policy():
    def __init__(self):
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.decay = 3000
        self.steps = 0
        self.action_space = [0, 1]

    def get_epsilon(self):
        epsilon = self.end_epsilon + (self.start_epsilon - self.end_epsilon) * \
            math.exp(-1. * self.steps / self.decay)
        return epsilon

    def get_action(self, q_network, current_state):
        self.steps += 1
        if random.random() < self.get_epsilon():
            return random.choice(self.action_space)
        else:
            q_values = q_network(current_state)
            return torch.argmax(q_values, dim=-1).item()
