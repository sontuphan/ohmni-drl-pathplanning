import tensorflow as tf
from tensorflow import keras
from tf_agents.agents import dqn, reinforce
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.utils import common


class REINFORCE:
    def __init__(self, env):
        self.env = env
        self.net = actor_distribution_network.ActorDistributionNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(64, 2))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-3)

    def gen_agent(self, train_step_counter):
        agent = reinforce.reinforce_agent.ReinforceAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            actor_network=self.net,
            optimizer=self.optimizer,
            normalize_returns=True,
            train_step_counter=train_step_counter)
        agent.initialize()
        return agent


class DQN_FC:
    def __init__(self, env):
        self.env = env
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(64,)
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


class DQN_CNN:
    def __init__(self, env):
        self.env = env
        self.input_shape = (96, 96)
        self.extractor = keras.applications.MobileNetV2(
            input_shape=(self.input_shape+(3,)),
            include_top=False,
            weights='imagenet'
        )
        self.extractor.trainable = False
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            preprocessing_layers=self.extractor,
            fc_layer_params=(64,)
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
