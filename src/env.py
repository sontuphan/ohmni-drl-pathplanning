import pyvirtualdisplay
from tf_agents.environments import suite_gym, tf_py_environment


class CartPole:
    def __init__(self, virtual=False):
        if virtual:
            pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        self.name = 'CartPole-v0'

    def gen_env(self):
        display = suite_gym.load(self.name)
        env = tf_py_environment.TFPyEnvironment(display)
        return env, display
