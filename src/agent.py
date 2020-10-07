import tensorflow as tf
from tf_agents.agents import dqn, reinforce
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.experimental.train.utils import strategy_utils


class REINFORCE:
    def __init__(self, env):
        self.env = env
        self.net = actor_distribution_network.ActorDistributionNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            conv_layer_params=[(32, 5, 1), (64, 5, 2),
                               (128, 5, 2), (256, 5, 2)],
            fc_layer_params=(64, 2))
        self.optimizer = tf.compat.v1.train.AdamOptimizer()
        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    def gen_agent(self, train_step_counter):
        with self.strategy.scope():
            agent = reinforce.reinforce_agent.ReinforceAgent(
                self.env.time_step_spec(),
                self.env.action_spec(),
                actor_network=self.net,
                optimizer=self.optimizer,
                normalize_returns=True,
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


class DQN:
    def __init__(self, env):
        self.env = env
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            conv_layer_params=[(32, 5, 1), (64, 5, 2),
                               (128, 5, 2), (256, 5, 2)],
            fc_layer_params=(128, 64)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer()
        self.strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    def gen_agent(self, train_step_counter):
        with self.strategy.scope():
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
