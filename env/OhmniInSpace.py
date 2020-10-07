import pybullet as p
import pybullet_data
import numpy as np
import cv2 as cv

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from env.objs import floor, ohmni, obstacle

VELOCITY_COEFFICIENT = 15
THROTTLE_RANGE = [-1, 1]
INTERPRETER = [[-0.4, -0.4], [-0.4, 0.4], [0., 0.], [0.4, -0.4], [0.4, 0.4]]


class Env:
    def __init__(self, gui=False, num_of_obstacles=4, image_shape=(96, 96)):
        self.gui = gui
        self.timestep = 0.05
        self.num_of_obstacles = num_of_obstacles
        self.image_shape = image_shape
        self.clientId = self._init_ws()
        self._left_wheel_id = 0
        self._right_wheel_id = 1

        # Start for the first time
        self._reset()

    def _init_ws(self):
        """
        Create server and start, there are two modes:
        1. GUI: it visualizes the environment and allow controlling
            ohmni via sliders.
        2. Headless: by running everything in background, it's suitable
            for ai/ml/rl development.
        """
        # Init server
        clientId = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=clientId)
        p.setTimeStep(self.timestep, physicsClientId=clientId)

        # Return
        return clientId

    def _build(self):
        """ Including floor, ohmni, obstacles into the environment """
        # Add gravity
        p.setGravity(0, 0, -10, physicsClientId=self.clientId)
        # Add plane and ohmni
        floor(self.clientId, texture=False, wall=False)
        ohmniId, _capture_image = ohmni(self.clientId)
        # Add obstacles at random positions
        for _ in range(self.num_of_obstacles):
            obstacle(self.clientId)
        # Return
        return ohmniId, _capture_image

    def _reset(self):
        """ Remove all objects, then rebuild them """
        p.resetSimulation(physicsClientId=self.clientId)
        self.ohmniId, self._capture_image = self._build()

    def capture_image(self):
        """ Get image from navigation camera """
        if self._capture_image is None:
            raise ValueError('_capture_image is undefined')
        return self._capture_image(self.image_shape)

    def getContactPoints(self):
        """ Get Ohmni contacts """
        return p.getContactPoints(self.ohmniId, physicsClientId=self.clientId)

    def getBasePositionAndOrientation(self):
        """ Get Ohmni position and orientation """
        return p.getBasePositionAndOrientation(self.ohmniId, physicsClientId=self.clientId)

    def reset(self):
        """ Reset the environment """
        self._reset()

    def step(self, action):
        """ Controllers for left/right wheels which are separate """
        # Normalize velocities
        [left_wheel, right_wheel] = INTERPRETER[action]
        left_wheel = left_wheel*VELOCITY_COEFFICIENT
        right_wheel = right_wheel*VELOCITY_COEFFICIENT
        # Step
        p.setJointMotorControl2(self.ohmniId, self._left_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel,
                                physicsClientId=self.clientId)
        p.setJointMotorControl2(self.ohmniId, self._right_wheel_id,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel,
                                physicsClientId=self.clientId)
        p.stepSimulation(physicsClientId=self.clientId)


class PyEnv(py_environment.PyEnvironment):
    def __init__(self, gui=False, image_shape=(96, 96)):
        super(PyEnv, self).__init__()
        # Parameters
        self.image_shape = image_shape
        self._image_dim = self.image_shape + (3,)
        # Self-defined variables
        self._num_of_obstacles = 5
        self._destination = np.array([10, 0, 0], dtype=np.float32)
        self._max_steps = 500
        self._num_steps = 0
        # PyEnvironment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int64,
            minimum=0,
            maximum=4,
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._image_dim, dtype=np.float32,
            minimum=np.zeros(self._image_dim, dtype=np.float32),
            maximum=np.full(self._image_dim, 1, dtype=np.float32),
            name='observation'
        )
        self._state = np.zeros(self._image_dim, dtype=np.float32)
        self._img = np.zeros(self._image_dim, dtype=np.float32)
        self._episode_ended = False
        self._discount = 0.9
        # Init bullet server
        self._env = Env(
            gui,
            num_of_obstacles=self._num_of_obstacles,
            image_shape=self.image_shape
        )

    def _normalized_distance_to_destination(self):
        """ Compute the distance from agent to destination """
        position, _ = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        distance = np.linalg.norm(position-self._destination)
        origin = np.linalg.norm(np.zeros([3])-self._destination)
        return min(distance/origin, 1)

    def _is_fatal(self):
        """ Compute whether there are collisions or not """
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        collision = self._env.getContactPoints()
        for contact in collision:
            # Contact with things different from floor
            if contact[2] != 0:
                return True
        # Ohmni felt out of the environment
        if position[2] >= 0.5 or position[2] <= -0.5:
            return True
        # Ohmni is falling down
        if abs(orientation[0]) > 0.2 or abs(orientation[1]) > 0.2:
            return True
        return False

    def _compute_reward(self):
        """ Compute reward and return (<stopped>, <reward>) """
        normalized_distance = self._normalized_distance_to_destination()
        # Reward shaping
        shaped_reward = 1 - normalized_distance
        # Ohmni reach the destination
        if normalized_distance < 0.1:
            return True, shaped_reward + (self._max_steps-self._num_steps)
        # Stop if detecting collisions or a fall
        if self._is_fatal():
            return True, shaped_reward - 1
        # Ohmni on his way
        return False, shaped_reward

    def _reset(self):
        """ Reset """
        self._num_steps = 0
        self._episode_ended = False
        self._env.reset()
        return ts.restart(np.zeros(self._image_dim, dtype=np.float32))

    def _step(self, action):
        """ Step, action is velocities of left/right wheel """
        # Reset if ended
        if self._episode_ended:
            return self.reset()
        self._num_steps += 1
        # Step the environment
        self._env.step(action)
        # Compute and save states
        _, _, rgb_img, _, seg_img = self._env.capture_image()
        self._img = np.array(rgb_img, dtype=np.float32)/255
        mask = np.minimum(seg_img, 1, dtype=np.float32)
        self._state = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        self._episode_ended, reward = self._compute_reward()
        # If exceed the limitation of steps, return rewards
        if self._num_steps > self._max_steps:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        # Transition
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward, discount=self._discount)

    def action_spec(self):
        """ Return action specs """
        return self._action_spec

    def observation_spec(self):
        """ Return observation specs """
        return self._observation_spec

    def render(self, mode='rgb_array'):
        """ Show video stream from navigation camera """
        img = cv.cvtColor(self._img, cv.COLOR_RGB2BGR)
        img = cv.resize(img, (512, 512))
        cv.imshow('Navigation view', img)
        cv.waitKey(10)
        return self._img


class TfEnv():
    def __init__(self):
        self.name = 'OhmniInSpace-v0'

    def gen_env(self, gui=False):
        """ Convert pyenv to tfenv """
        pyenv = PyEnv(gui=gui)
        tfenv = tf_py_environment.TFPyEnvironment(pyenv)
        return tfenv
