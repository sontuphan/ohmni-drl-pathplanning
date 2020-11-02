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
        self._num_actions = 5
        self._num_step = 0
        # Model
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
            keras.layers.Dense(self._num_actions),
        ])
        self.optimizer = keras.optimizers.Adam()
        # Setup checkpoints
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              net=self.model)
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, CHECKPOINT_DIR, max_to_keep=1)
        self.checkpoint.restore(self.manager.latest_checkpoint)

    def _define_collect_data_spec(self, env):
        return trajectory.from_transition(
            env.time_step_spec(),
            policy_step.PolicyStep(action=env.action_spec()),
            env.time_step_spec(),
        )

    def epsilon(self):
        return 0.1 * (1 - tf.exp(-0.00001 * self._num_step))

    def explore(self, actions):
        print('Step {} / Epsilon {}', self._num_step, self.epsilon())
        _epsilons = tf.cast(
            tf.greater(
                tf.random.uniform(actions.shape, minval=0, maxval=1),
                tf.fill(actions.shape, self.epsilon()),
            ),
            dtype=tf.int32
        )
        _random_actions = tf.random.uniform(
            actions.shape, minval=0, maxval=4, dtype=tf.int32)
        _actions = _epsilons*_random_actions + (1-_epsilons)*actions
        return _actions

    def action(self, _time_step):
        self._num_step += 1
        _qvalues = self.model(_time_step.observation)
        _actions = tf.argmax(_qvalues, axis=1, output_type=tf.int32)
        _actions = self.explore(_actions)
        return policy_step.PolicyStep(action=_actions, state=_qvalues)

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
            self.manager.save()
        return loss
