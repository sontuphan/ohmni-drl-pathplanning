import tensorflow as tf
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

from src.env import CartPole
from src.agent import DQN
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Compulsory config for tf_agents
tf.compat.v1.enable_v2_behavior()

# Environment
cartpole = CartPole()
train_env, train_display = cartpole.gen_env()
eval_env, eval_display = cartpole.gen_env()

# Agent
dqn = DQN(train_env)
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
for step in range(100):
    replay_buffer.collect(train_env, random_policy, step)
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

num_iterations = 200
collect_steps_per_iteration = 100
for _ in range(num_iterations):
    # Collect a few steps using collect_policy and save to the replay buffer.
    # You may ask why do not I see any env.reset() here
    # The unspoken info that the step() of tf_env will automatically
    # reset the env if it meets LAST_STATE
    # (refer https://www.tensorflow.org/agents/api_docs/python/tf_agents/environments/tf_py_environment/TFPyEnvironment#step)
    for step in range(collect_steps_per_iteration):
        replay_buffer.collect(train_env, agent.collect_policy, step)
    # Sample a batch of data from the buffer and update the agent's network.
    experience, unused_info = next(database)
    train_loss = agent.train(experience).loss
    step = agent.train_step_counter.numpy()
    if step % 100 == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))
    if step % 1000 == 0:
        avg_return = criterion.eval(eval_env, agent.policy)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))

# Visualization
criterion.display()
# criterion.record(eval_env, eval_display, agent.policy, "trained-agent")
