import gym
import torch
from collections import namedtuple
import random
import numpy as np
import matplotlib.pyplot as plt
import math


MODEL_PATH = './model'
GAMMA = 0.9  # Discount
BATCH_SIZE = 64
STEP_PER_EPOCH = 1024
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(4, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        self.loss_func = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, x):
        return self.model(x)


class ReplayMemory(object):
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.capacity = self.batch_size*16
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def isFull(self):
        return len(self.memory) >= self.capacity


class Policy():
    def __init__(self):
        self.start_epsilon = 1
        self.end_epsilon = 0.01
        self.decay = 6000
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


def plot_durations(durations, means, epsilons):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    plt.plot(durations, label="Duration")
    plt.plot(means, label="Mean")
    plt.legend()

    plt.twinx()
    plt.ylabel('Epsilon')
    plt.plot(epsilons, '#48a868')
    
    plt.pause(0.001)


if __name__ == "__main__":

    # Define networks
    policy_net = NeuralNetwork()
    target_net = NeuralNetwork()
    target_net.load_state_dict(policy_net.state_dict())

    # Define policy
    policy = Policy()

    # Define environment
    replaymem = ReplayMemory(BATCH_SIZE)
    env = gym.make("CartPole-v1")
    state = env.reset()

    # Controllers
    duration = 0
    steps = 0
    episode_durations = np.array([])
    episode_means = np.array([])
    episode_epsilons = np.array([])

    while True:
        duration += 1
        steps += 1
        env.render()
        x = torch.from_numpy(state).float()
        action = policy.get_action(policy_net, x)
        next_state, reward, done, info = env.step(action)
        replaymem.push(state, action, next_state, reward)

        # Train
        if replaymem.isFull():
            # Get a batch of data
            transitions = replaymem.sample()
            batch = Transition(*zip(*transitions))
            (batch_states, batch_actions, batch_next_states, batch_rewards) = batch
            batch_states = torch.from_numpy(
                np.array(batch_states, dtype=np.float64)).float()
            batch_actions = torch.from_numpy(
                np.array(batch_actions, dtype=np.float64)).float()
            batch_next_states = torch.from_numpy(
                np.array(batch_next_states, dtype=np.float64)).float()
            batch_rewards = torch.from_numpy(
                np.array(batch_rewards, dtype=np.float64)).float()
            # Predict Q values
            batch_pred_q_values = torch.max(
                policy_net(batch_states), dim=-1).values
            # Get Q values label
            batch_q_values = torch.max(target_net(
                batch_next_states), dim=-1).values + GAMMA*batch_rewards
            # Calculate loss
            loss = policy_net.loss_func(
                batch_pred_q_values, batch_q_values)
            # Gradient acsend
            policy_net.optimizer.zero_grad()
            loss.backward()
            policy_net.optimizer.step()

        if steps % STEP_PER_EPOCH == 0:
            target_net.load_state_dict(policy_net.state_dict())
            steps = 0

        state = next_state
        if reward == 0:
            episode_durations = np.append(episode_durations, duration)
            episode_epsilons = np.append(
                episode_epsilons, policy.get_epsilon())
            if len(episode_durations) < 101:
                episode_means = np.append(episode_means, 0)
            else:
                episode_means = np.append(
                    episode_means, np.mean(episode_durations[-101:-1]))
            plot_durations(episode_durations, episode_means, episode_epsilons)
            duration = 0
            state = env.reset()

    env.close()
