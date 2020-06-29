import gym
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import torch

from src.nets import ConvNet, MobileNet
from src.mem import ReplayMemory
from src.policy import Policy

MODEL_PATH = './model'
GAMMA = 0.9  # Discount
BATCH_SIZE = 64
STEP_PER_EPOCH = 1024


def convert_cv_to_pil(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


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
    policy_net = MobileNet()
    target_net = MobileNet()
    target_net.load_state_dict(policy_net.state_dict())

    # Define policy
    policy = Policy()

    # Define environment
    replaymem = ReplayMemory(BATCH_SIZE)
    env = gym.make("CartPole-v1")
    env.reset()

    # Controllers
    duration = 0
    steps = 0
    episode_durations = np.array([])
    episode_means = np.array([])
    episode_epsilons = np.array([])

    while True:
        duration += 1
        state = env.render(mode='rgb_array')
        state = policy_net.normalize_data(state)
        x = torch.unsqueeze(state, 0)
        action = policy.get_action(policy_net, x)
        _, reward, _, _ = env.step(action)
        next_state = env.render(mode='rgb_array')
        next_state = policy_net.normalize_data(next_state)
        # Add observation to memory
        replaymem.push(state.numpy(), action, next_state.numpy(), reward)

        # Train
        if replaymem.isFull():
            # Increase step
            steps += 1
            # Get a batch of data
            transitions = replaymem.sample()
            batch = replaymem.Transition(*zip(*transitions))
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
            batch_predicted_q_values = torch.max(
                policy_net(batch_states), dim=-1).values
            # Get ground-truth Q values
            batch_expected_q_values = GAMMA*torch.max(target_net(
                batch_next_states), dim=-1).values + batch_rewards
            # Calculate loss
            loss = policy_net.criterion(
                batch_predicted_q_values, batch_expected_q_values)
            # Gradient acsend
            policy_net.optimizer.zero_grad()
            loss.backward()
            policy_net.optimizer.step()
            # Update target network
            if steps % STEP_PER_EPOCH == 0:
                print('*** Updated target network:',
                      torch.mean(batch_expected_q_values))
                target_net.load_state_dict(policy_net.state_dict())
                steps = 0

        # Update env
        if reward == 0:
            episode_durations = np.append(episode_durations, duration)
            episode_epsilons = np.append(
                episode_epsilons, policy.get_epsilon())
            if len(episode_durations) > 1024:
                episode_means = np.append(
                    episode_means, np.mean(np.array(episode_durations)[-1024: -1]))
            else:
                episode_means = np.append(episode_means, 0)
            plot_durations(episode_durations, episode_means, episode_epsilons)
            duration = 0
            state = env.reset()

    env.close()
