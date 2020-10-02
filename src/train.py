import os
import tensorflow as tf
from tf_agents.utils import common

from env import OhmniInSpace
from src.agent import REINFORCE
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Saving dir
saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()


def train():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    train_env = tfenv.gen_env()
    eval_env = tfenv.gen_env()

    # Agent
    algo = REINFORCE(train_env)
    train_step_counter = tf.Variable(0)
    agent = algo.gen_agent(train_step_counter)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
    )

    # Metrics and Evaluation
    criterion = ExpectedReturn()

    # Train
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    criterion.eval(eval_env, agent.policy)

    num_iterations = 100
    for _ in range(num_iterations):
        replay_buffer.collect_episode(train_env, agent.collect_policy, 2)
        experience = replay_buffer.buffer.gather_all()
        train_loss = agent.train(experience).loss
        replay_buffer.buffer.clear()
        step = agent.train_step_counter.numpy()
        if step % 1 == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))
        if step % 10 == 0:
            avg_return = criterion.eval(eval_env, agent.policy)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    # Visualization
    criterion.save()
    REINFORCE.save_policy(agent.policy, saving_dir)


def run():
    tfenv = OhmniInSpace.TfEnv()
    env = tfenv.gen_env(gui=True)
    policy = REINFORCE.load_policy(saving_dir)
    time_step = env.reset()
    while True:
        action_step = policy.action(time_step)
        print('Action:', action_step)
        time_step = env.step(action_step.action)
