import tensorflow as tf
from tensorflow import keras
from tf_agents.trajectories import policy_step, trajectory
from tf_agents.policies import random_tf_policy


class DQN():
    def __init__(self, env):
        self.collect_data_spec = self._define_collect_data_spec(env)
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

    def _define_collect_data_spec(self, env):
        return trajectory.from_transition(
            env.time_step_spec(),
            policy_step.PolicyStep(action=env.action_spec()),
            env.time_step_spec(),
        )

    def action(self, time_step):
        _qvalues = self.model(time_step.observation)
        _action = tf.argmax(_qvalues, axis=1, output_type=tf.int32)
        return policy_step.PolicyStep(action=_action, state=_qvalues)

    def train_step(self, qs, q_targets):
        print(qs, q_targets)
        return None

    def train(self, experience):
        states, next_states = tf.split(experience.observation,
                       num_or_size_splits=[1, 1], axis=1)
        q_values = self.model(tf.squeeze(states))
        next_q_values = self.model(tf.squeeze(next_states))
        print(q_values, next_q_values)
        return self.train_step(1, 2)
