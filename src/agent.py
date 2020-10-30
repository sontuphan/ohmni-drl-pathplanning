import os
import tensorflow as tf
from tensorflow import keras
from tf_agents.trajectories import policy_step, trajectory, time_step


# Saving dir
policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')


class DQN():
    def __init__(self, env):
        # Params
        self.collect_data_spec = self._define_collect_data_spec(env)
        self.discount = 0.99
        # Model
        with tf.distribute.MirroredStrategy().scope():
            self.model = keras.models.Sequential([  # (96, 96, *)
                keras.layers.Conv2D(  # (92, 92, 16)
                    filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                    input_shape=(96, 96, 4)),
                keras.layers.MaxPooling2D((2, 2)),  # (46, 46, 16)
                keras.layers.Conv2D(  # (42, 42, 32)
                    filters=32, kernel_size=(5, 5), strides=(1, 1), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),  # (21, 21, 32)
                keras.layers.Conv2D(  # (10, 10, 64)
                    filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),  # (5, 5, 64)
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(5),
            ])
        self.optimizer = keras.optimizers.Adam()
        # Setup checkpoints
        self.checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              net=self.model)
        # self.checkpoint.restore(
        #     tf.train.latest_checkpoint(self.checkpoint_dir))

    def _define_collect_data_spec(self, env):
        return trajectory.from_transition(
            env.time_step_spec(),
            policy_step.PolicyStep(action=env.action_spec()),
            env.time_step_spec(),
        )

    def action(self, _time_step):
        _qvalues = self.model(_time_step.observation)
        _action = tf.argmax(_qvalues, axis=1, output_type=tf.int32)
        return policy_step.PolicyStep(action=_action, state=_qvalues)

    @tf.function
    def train_step(self, step_types, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            q_values = tf.gather_nd(self.model(states), actions, batch_dims=1)
            next_q_values = tf.reduce_max(self.model(next_states), axis=1)
            step_types = tf.cast(
                tf.less(step_types, time_step.StepType.LAST), dtype=tf.float32)
            q_targets = rewards + self.discount*next_q_values*step_types
            loss = tf.reduce_sum(tf.square(q_values-q_targets))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, experience, step):
        states, next_states = tf.squeeze(tf.split(experience.observation,
                                                  num_or_size_splits=[1, 1], axis=1))
        actions, _ = tf.split(experience.action,
                              num_or_size_splits=[1, 1], axis=1)
        rewards, _ = tf.squeeze(tf.split(experience.reward,
                                         num_or_size_splits=[1, 1], axis=1))
        step_types, _ = tf.squeeze(tf.split(experience.step_type,
                                            num_or_size_splits=[1, 1], axis=1))
        loss = self.train_step(
            step_types, states, actions, rewards, next_states)
        if step % 1000 == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        return loss
