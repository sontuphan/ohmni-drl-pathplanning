import tensorflow as tf
from tensorflow import keras


class DQN():
    def __init__(self, env, train_step_counter):
        self.train_step_counter = train_step_counter
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
            keras.layers.Dense(5, activation='relu'),
        ])
        self.optimizer = keras.optimizers.Adam()
    
    def _define_collect_data_spec(self, env):
        print(env.action_spec)
        print(env.observation_spec)
        return None


    @tf.function
    def train_step(self, ds):
        self.train_step_counter += 1

    def train(self, ds):
        pass
