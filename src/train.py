import time
import tensorflow as tf
import cv2 as cv
import numpy as np
from tf_agents.utils import common

from env import OhmniInSpace
from src.agent import DQN
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: tranning servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0


def run():
    # # Environment
    # tfenv = OhmniInSpace.TfEnv()
    # env = tfenv.gen_env(gui=LOCAL)
    # # Agent
    # agent = DQN(env, training=False)
    # while True:
    #     time_step = env.current_time_step()
    #     observation = np.squeeze(time_step.observation.numpy())
    #     cv.imshow('Segmentation', observation)
    #     if cv.waitKey(10) & 0xFF == ord('q'):
    #         break
    #     action_step = agent.action(time_step)
    #     env.step(action_step.action)
    pass


def train():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    train_env = tfenv.gen_env(gui=LOCAL)
    eval_env = tfenv.gen_env()

    # Agent
    algo = DQN(train_env)
    train_step_counter = tf.Variable(0)
    agent = algo.gen_agent(train_step_counter)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        sample_batch_size=8,
    )

    # Metrics and Evaluation
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    criterion = ExpectedReturn()

    # Train
    num_iterations = 1000000
    eval_step = 5000
    start = time.time()
    loss = 0
    replay_buffer.collect_step(train_env, agent.collect_policy)
    dataset = replay_buffer.pipeline()
    while agent.train_step_counter.numpy() <= num_iterations:
        replay_buffer.collect_step(train_env, agent.collect_policy)
        experience, _ = next(dataset)
        loss += agent.train(experience).loss
        # Evaluation
        step = agent.train_step_counter.numpy()
        if step % eval_step == 0:
            # Checkpoints
            avg_return = criterion.eval(eval_env, agent.policy)
            print('Step = {0}: Average Return = {1} / Average Loss = {2}'.format(
                step, avg_return, loss/eval_step))
            end = time.time()
            print('Step estimated time: {:.4f}'.format((end-start)/eval_step))
            # Reset
            start = time.time()
            loss = 0

    # Visualization
    criterion.save()
