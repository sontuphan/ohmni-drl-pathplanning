import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

from env import OhmniInSpace
from src.agent import DQN_CNN
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()


def train():
    # Environment
    oit = OhmniInSpace.TfEnv()
    train_env, train_display = oit.gen_env()
    eval_env, eval_display = oit.gen_env()

    # Agent
    dqn = DQN_CNN(train_env)
    train_step_counter = tf.Variable(0)
    agent = dqn.gen_agent(train_step_counter)

    # Policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    # Replay buffer
    replay_buffer = ReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
    )
    # Initial data
    for _ in range(100):
        replay_buffer.collect(train_env, random_policy)
    # Create dataset
    database = replay_buffer.get_pipeline(
        num_parallel_calls=3,
        num_steps=2,
        num_prefetch=3
    )

    # Metrics and Evaluation
    criterion = ExpectedReturn()

    # Train
    agent.train = common.function(agent.train)
    agent.train_step_counter.assign(0)
    criterion.eval(eval_env, agent.policy)

    num_iterations = 20000
    collect_steps_per_iteration = 1
    for _ in range(num_iterations):
        for _ in range(collect_steps_per_iteration):
            replay_buffer.collect(train_env, agent.collect_policy)
        experience, unused_info = next(database)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()
        if step % (num_iterations/50) == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))
        if step % (num_iterations/10) == 0:
            avg_return = criterion.eval(eval_env, agent.policy)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    # Visualization
    criterion.display()
    criterion.record(eval_env, eval_display, agent.policy, "trained-agent")
