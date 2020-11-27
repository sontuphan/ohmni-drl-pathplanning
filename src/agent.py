import tensorflow as tf
from tf_agents.agents import dqn
from tf_agents.networks import q_network
from tf_agents.utils import common
from tf_agents.policies import policy_saver


class DQN():
    def __init__(self, env, checkpoint_dir):
        # Env
        self.env = env
        # Policy
        self.net = q_network.QNetwork(
            self.env.observation_spec(),
            self.env.action_spec(),
            fc_layer_params=(128,)
        )
        # Agent
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00025)
        self.agent = dqn.dqn_agent.DqnAgent(
            self.env.time_step_spec(),
            self.env.action_spec(),
            q_network=self.net,
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

    @staticmethod
    def save_policy(policy, saving_dir):
        saver = policy_saver.PolicySaver(policy)
        saver.save(saving_dir)

    @staticmethod
    def load_policy(saving_dir):
        policy = tf.compat.v2.saved_model.load(saving_dir)
        return policy
