import os
import tensorflow as tf
from tensorflow import keras
from tf_agents.trajectories import policy_step, trajectory, time_step
import cv2 as cv


# Saving dir
policy_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../models/policy')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              '../models/checkpoints')


class DQN():
    def __init__(self, env, training=True):
        # Params
        self.collect_data_spec = self._define_collect_data_spec(env)
        self.discount = 0.99
        self._num_actions = 5
        self.step = tf.Variable(initial_value=0, dtype=tf.float32, name='step')
        self.training = training
        # Model
        self.policy = keras.Sequential([
            tf.keras.applications.MobileNetV2(input_shape=env.observation_spec().shape,
                                              include_top=False,
                                              weights='imagenet'),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu', name='attention_layer'),
            keras.layers.Dense(192, activation='relu', name='attention_layer'),
            keras.layers.Dense(self._num_actions, name='action_layer'),
        ])
        self.policy.layers[0].trainable = False
        self.policy.summary()
        self.optimizer = keras.optimizers.Adam()
        # Setup checkpoints
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.policy,
            step=self.step,
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, CHECKPOINT_DIR, max_to_keep=1)
        self.checkpoint.restore(self.manager.latest_checkpoint)
        # Debug
        self.extractor = keras.Model(
            inputs=self.policy.inputs,
            outputs=self.policy.get_layer(name='attention_layer').output
        )

    def _define_collect_data_spec(self, env):
        return trajectory.from_transition(
            env.time_step_spec(),
            policy_step.PolicyStep(action=env.action_spec()),
            env.time_step_spec(),
        )

    def _epsilon(self):
        if not self.training:
            return tf.constant(1, dtype=tf.float32)
        return 0.9 - tf.exp(-0.0001 * self.step)

    def get_step(self):
        return int(self.step.numpy())

    def increase_step(self):
        self.step.assign_add(1)

    def explore(self, actions):
        _epsilons = tf.cast(
            tf.greater(
                tf.random.uniform(actions.shape, minval=0, maxval=1),
                tf.fill(actions.shape, self._epsilon()),
            ),
            dtype=tf.int32
        )
        _random_actions = tf.random.uniform(
            actions.shape, minval=0, maxval=4, dtype=tf.int32)
        _actions = _epsilons*_random_actions + (1-_epsilons)*actions
        return _actions

    def pay_attention(self, observation):
        v = self.extractor(observation)
        v = tf.squeeze(v)
        mean, variance = tf.nn.moments(v, axes=[0])
        v = (v - mean)/tf.sqrt(variance)
        v = tf.reshape(v, [8, 8, 3])
        img = v.numpy()
        cv.imshow('Attention matrix', img)
        cv.waitKey(10)

    def action(self, _time_step):
        _qvalues = self.policy(_time_step.observation)
        if not self.training:
            self.pay_attention(_time_step.observation)
            print("Q values:", _qvalues.numpy())
        _actions = tf.argmax(_qvalues, axis=1, output_type=tf.int32)
        _actions = self.explore(_actions)
        return policy_step.PolicyStep(action=_actions, state=_qvalues)

    @tf.function
    def train_step(self, step_types, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            q_values = tf.gather_nd(self.policy(states), actions, batch_dims=1)
            next_q_values = tf.reduce_max(self.policy(next_states), axis=1)
            step_types = tf.cast(
                tf.less(step_types, time_step.StepType.LAST), dtype=tf.float32)
            q_targets = rewards + self.discount*next_q_values*step_types
            loss = tf.reduce_sum(tf.square(q_values-q_targets))
        variables = self.policy.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def train(self, experience):
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
        if self.step % 1000 == 0:
            self.manager.save()
        return loss
