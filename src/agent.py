import os
import tensorflow as tf
from tf_agents.agents import dqn
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.policies import policy_saver

# Saving dir
policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')


class DQN():
    def __init__(self, env):
        self.env = env
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(128,)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-3)

    def gen_agent(self, train_step_counter):
        agent = dqn.dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        agent.initialize()
        return agent

    @staticmethod
    def save_policy(policy, saving_dir):
        saver = policy_saver.PolicySaver(policy)
        saver.save(saving_dir)

    @staticmethod
    def load_policy(saving_dir):
        policy = tf.compat.v2.saved_model.load(saving_dir)
        return policy
