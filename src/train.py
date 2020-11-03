import time
import tensorflow as tf

from env import OhmniInSpace
from src.agent import DQN
from src.buffer import ReplayBuffer
from src.eval import ExpectedReturn

# Trick
# No GPU: my super-extra-fast-and-furiuos-ahuhu machine
# GPUs: tranning servers
LOCAL = not len(tf.config.list_physical_devices('GPU')) > 0


def run():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    env = tfenv.gen_env(gui=LOCAL)
    # Agent
    agent = DQN(env, training=False)
    while True:
        time_step = env.current_time_step()
        action_step = agent.action(time_step)
        env.step(action_step.action)


def train():
    # Environment
    tfenv = OhmniInSpace.TfEnv()
    train_env = tfenv.gen_env(gui=LOCAL)
    eval_env = tfenv.gen_env()

    # Agent
    agent = DQN(train_env)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        sample_batch_size=8,
    )
    replay_buffer.collect_step(train_env, agent)
    dataset = replay_buffer.pipeline()

    # Metrics and Evaluation
    criterion = ExpectedReturn()

    # Train
    num_iterations = 1000000
    eval_step = 1000
    start = time.time()
    for _ in range(num_iterations):
        agent.increase_step()
        replay_buffer.collect_step(train_env, agent)
        experience, _ = next(dataset)
        agent.train(experience)
        # Evaluation
        if agent.step % eval_step == 0:
            # Checkpoints
            avg_return = criterion.eval(eval_env, agent)
            print('Step = {0}: Average Return = {1}'.format(
                agent.step, avg_return))
            end = time.time()
            print('Step estimated time: {:.4f}'.format((end-start)/eval_step))
            start = time.time()

    # Visualization
    criterion.save()
