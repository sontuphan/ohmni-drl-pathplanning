import os
import time
import tensorflow as tf
import cv2 as cv
import numpy as np
from tf_agents.utils import common

from env import OhmniInSpace
from src.agent import PPO
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: tranning servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0

# Saving dir
POLICY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')


def run():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    env = tfenv.gen_env(gui=LOCAL)
    # Agent
    ppo = PPO(env, CHECKPOINT_DIR)
    ppo.load_checkpoint()
    while True:
        time_step = env.current_time_step()
        # observation = np.squeeze(time_step.observation.numpy())
        # cv.imshow('Segmentation', observation)
        # if cv.waitKey(10) & 0xFF == ord('q'):
        #     break
        action_step = ppo.agent.policy.action(time_step)
        print('Acction:', action_step.action.numpy())
        env.step(action_step.action)


def train():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    train_env = tfenv.gen_env(gui=LOCAL)
    eval_env = tfenv.gen_env()

    # Agent
    ppo = PPO(train_env, CHECKPOINT_DIR)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        ppo.agent.collect_data_spec,
        batch_size=train_env.batch_size,
        epochs=1,
    )

    # Metrics and Evaluation
    ppo.agent.train = common.function(ppo.agent.train)
    criterion = ExpectedReturn()

    # Train
    num_iterations = 1000000
    eval_step = 5000
    start = time.time()
    loss = 0
    while ppo.agent.train_step_counter.numpy() <= num_iterations:
        print(ppo.agent.train_step_counter.numpy())
        replay_buffer.collect_episode(train_env, ppo.agent.collect_policy)
        experience = replay_buffer.gather_all()
        loss += ppo.agent.train(experience).loss
        replay_buffer.clear()
        # Evaluation
        step = ppo.agent.train_step_counter.numpy()
        if step % eval_step == 0:
            # Checkpoints
            ppo.save_checkpoint()
            # Evaluation
            avg_return = criterion.eval(eval_env, ppo.agent.policy)
            print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
                step, avg_return, loss/eval_step))
            end = time.time()
            print('Step estimated time: {:.4f}'.format((end-start)/eval_step))
            # Reset
            start = time.time()
            loss = 0

    # Visualization
    criterion.save()
