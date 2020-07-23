import tensorflow as tf
from tensorflow import keras
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common


class DQN_FC:
    def __init__(self, env):
        self.env = env
        self.q_net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(64,)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-3)

    def gen_agent(self, train_step_counter):
        agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.q_net,
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
        self.q_net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            preprocessing_layers=self.extractor,
            fc_layer_params=(64,)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=1e-3)

    def gen_agent(self, train_step_counter):
        agent = dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.q_net,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=train_step_counter)
        agent.initialize()
        return agent
