import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_network
from tf_agents.utils import common


class DQN:
    def __init__(self, env):
        self.env = env
        self.learning_rate = 1e-3
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
