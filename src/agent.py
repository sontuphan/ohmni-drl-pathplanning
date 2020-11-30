import tensorflow as tf
from tf_agents.agents import ppo
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.utils import common


class DQN():
    def __init__(self, env, checkpoint_dir):
        # Env
        self.env = env
        # Policy
        self.actor = actor_distribution_network.ActorDistributionNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(128,)
        )
        self.critic = value_network.ValueNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(128,)
        )
        # Agent
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=0.000025)
        self.agent = ppo.ppo_clip_agent.PPOClipAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            actor_net=self.actor,
            value_net=self.critic,
            optimizer=self.optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            train_step_counter=self.global_step)
        self.agent.initialize()
        # Checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.global_step
        )

    def save_checkpoint(self):
        self.checkpointer.save(self.global_step)

    def load_checkpoint(self):
        self.checkpointer.initialize_or_restore()
        self.global_step = tf.compat.v1.train.get_global_step()
