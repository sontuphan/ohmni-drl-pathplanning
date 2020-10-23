import os
import time
import tensorflow as tf
from tf_agents.utils import common

from env import OhmniInSpace
from src.agent import DQN
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Saving dir
saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')
# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Trick
# No GPU: my super-extra-fast-and-furiuos-huhu machine
# GPUs: tranning servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0


def check_point():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    train_env = tfenv.gen_env(gui=LOCAL)

    # Agent
    algo = DQN(train_env)
    train_step_counter = tf.Variable(0)
    agent = algo.gen_agent(train_step_counter)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
    )

    # Train
    agent.train_step_counter.assign(0)
    num_iterations = 100000
    algo.load_checkpoint(checkpoint_dir, agent, replay_buffer.buffer)
    for _ in range(num_iterations):
        replay_buffer.collect_step(train_env, agent.collect_policy)


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
    )
    dataset = replay_buffer.get_pipeline()

    # Metrics and Evaluation
    criterion = ExpectedReturn()

    # Train
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)

    num_iterations = 1000000
    eval_step = 1000
    algo.load_checkpoint(checkpoint_dir, agent, replay_buffer.buffer)
    start = time.time()
    for _ in range(num_iterations):
        replay_buffer.collect_step(train_env, agent.collect_policy)
        experience, _ = next(dataset)
        train_loss = agent.train(experience)
        step = agent.train_step_counter.numpy()
        # Evaluation
        if step % eval_step == 0:
            # Checkpoints
            algo.save_checkpoint(checkpoint_dir, agent, replay_buffer.buffer)
            avg_return = criterion.eval(eval_env, agent.policy)
            print('Step = {0}: Average Return = {1}'.format(step, avg_return))
            end = time.time()
            print('Step estimated time: {:.4f}'.format((end-start)/eval_step))
            start = time.time()

    # Visualization
    criterion.save()
    DQN.save_policy(agent.policy, saving_dir)


def run():
    tfenv = OhmniInSpace.TfEnv()
    env = tfenv.gen_env(gui=True)
    policy = DQN.load_policy(saving_dir)
    time_step = env.reset()
    while True:
        action_step = policy.action(time_step)
        print('Action:', action_step)
        time_step = env.step(action_step.action)
