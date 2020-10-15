from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow import keras
from tf_agents.agents import dqn, reinforce
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.experimental.train.utils import strategy_utils


class Agent(ABC):
    def __init__(self, env):
        self.env = env
        self.strategy = strategy_utils.get_strategy(
            tpu=False,
            use_gpu=(len(tf.config.list_physical_devices('GPU')) > 0)
        )
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.checkpointer = None

    def _checkpoint(self, checkpoint_dir, agent, replay_buffer):
        if self.checkpointer is None:
            self.checkpointer = common.Checkpointer(
                ckpt_dir=checkpoint_dir,
                max_to_keep=1,
                agent=agent,
                policy=agent.policy,
                replay_buffer=replay_buffer,
                global_step=self.global_step
            )
        return self.checkpointer

    def save_checkpoint(self, checkpoint_dir, agent, replay_buffer):
        checkpointer = self._checkpoint(checkpoint_dir, agent, replay_buffer)
        checkpointer.save(self.global_step)

    def load_checkpoint(self, checkpoint_dir, agent, replay_buffer):
        checkpointer = self._checkpoint(checkpoint_dir, agent, replay_buffer)
        checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()

    @staticmethod
    def save_policy(policy, saving_dir):
        saver = policy_saver.PolicySaver(policy)
        saver.save(saving_dir)

    @staticmethod
    def load_policy(saving_dir):
        policy = tf.compat.v2.saved_model.load(saving_dir)
        return policy

    @abstractmethod
    def gen_agent(self, train_step_counter):
        """ Generate an agent instance """
        return


class REINFORCE(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.net = actor_distribution_network.ActorDistributionNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            conv_layer_params=[(32, 5, 1), (64, 5, 2),
                               (128, 5, 2), (256, 5, 2)],
            fc_layer_params=(64, 2))
        self.optimizer = tf.compat.v1.train.AdamOptimizer()

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


class DQN(Agent):
    def __init__(self, env):
        super().__init__(env)
        self.preprocessing_layers = {
            'mask': keras.models.Sequential([  # (96, 96, 3)
                keras.layers.Conv2D(  # (92, 92, 16)
                    filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),  # (46, 46, 16)
                keras.layers.Conv2D(  # (42, 42, 32)
                    filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),  # (21, 21, 32)
                keras.layers.Conv2D(  # (10, 10, 64)
                    filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),  # (5, 5, 64)
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
            ]),
            'pose': keras.layers.Dense(16, activation='relu'),
        }
        self.preprocessing_combiner = keras.layers.Concatenate(axis=-1)
        self.net = q_network.QNetwork(
            input_tensor_spec=self.env.observation_spec(),
            action_spec=self.env.action_spec(),
            preprocessing_layers=self.preprocessing_layers,
            preprocessing_combiner=self.preprocessing_combiner,
            fc_layer_params=(64, 32)
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer()

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
