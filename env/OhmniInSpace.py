import pybullet as p
import pybullet_data
import numpy as np
import cv2 as cv
import pyvirtualdisplay

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from env.objs import floor, ohmni, obstacle

THROTTLE_RANGE = [-15, 15]
STEERING_RANGE = [0, 8]


class Env:
    def __init__(self, gui=False, num_of_obstacles=4, image_shape=(96, 96)):
        self.gui = gui
        self.timestep = 0.05
        self.num_of_obstacles = num_of_obstacles
        self.image_shape = image_shape
        self.clientId, self._get_velocities_from_gui = self._init_ws()
        self._left_wheel_id = 0
        self._right_wheel_id = 1

        # Start for the first time
        self._reset()

        if self.gui:
            self._start()

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
        # Set debug parameters for GUI mode
        throttle, steering = None, None
        if self.gui:
            throttle = p.addUserDebugParameter(
                'Throttle',
                THROTTLE_RANGE[0], THROTTLE_RANGE[1], 0,
                physicsClientId=clientId)
            steering = p.addUserDebugParameter(
                'Steering',
                STEERING_RANGE[0], STEERING_RANGE[1], 0,
                physicsClientId=clientId)

        # Utility
        def _get_velocities_from_gui():
            user_throttle = 0
            user_steering = 0
            if self.gui:
                user_throttle = p.readUserDebugParameter(
                    throttle, physicsClientId=clientId)
                user_steering = p.readUserDebugParameter(
                    steering, physicsClientId=clientId)
            return user_throttle, user_steering
        # Return
        return clientId, _get_velocities_from_gui

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

    def _start(self):
        """ This function is only called in gui mode """
        while True:
            _, _, _, _, seg_img = self.capture_image()
            throttle, steering = self.capture_velocities()
            left_wheel, right_wheel = throttle+steering, throttle-steering
            self.step(left_wheel, right_wheel)
            mask = np.minimum(seg_img, 1, dtype=np.float32)
            cv.imshow('Segmentation', mask)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    def _reset(self):
        """ Remove all objects, then rebuild them """
        p.resetSimulation(physicsClientId=self.clientId)
        self.ohmniId, self._capture_image = self._build()

    def capture_image(self):
        """ Get image from navigation camera """
        if self._capture_image is None:
            raise ValueError('_capture_image is undefined')
        return self._capture_image(self.image_shape)

    def capture_velocities(self):
        """ Get user's inputs for velo params from GUI """
        if self._get_velocities_from_gui is None:
            raise ValueError('_get_velocities_from_gui is undefined')
        return self._get_velocities_from_gui()

    def getContactPoints(self):
        """ Get Ohmni contacts """
        return p.getContactPoints(self.ohmniId, physicsClientId=self.clientId)

    def getBasePositionAndOrientation(self):
        """ Get Ohmni position and orientation """
        return p.getBasePositionAndOrientation(self.ohmniId, physicsClientId=self.clientId)

    def reset(self):
        """ Reset the environment """
        self._reset()

    def step(self, left_wheel, right_wheel):
        """ Controllers for left/right wheels which are separate """
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
        # Self-defined variables
        self.image_shape = image_shape
        self._image_dim = self.image_shape + (3,)
        # Steering: left_wheel_velocity - right_wheel_velocity
        self._num_of_obstacles = 5
        self._destination = np.array([10, 0, 0], dtype=np.float32)
        self._max_steps = 500
        self._num_steps = 0
        # PyEnvironment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=STEERING_RANGE[0],
            maximum=STEERING_RANGE[1],
            name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self._image_dim, dtype=np.float32,
            minimum=np.zeros(self._image_dim, dtype=np.float32),
            maximum=np.zeros(self._image_dim, dtype=np.float32)+1,
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

    def _compute_reward(self):
        """ Compute reward and return (<stopped>, <reward>) """
        position, _ = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        distance = np.linalg.norm(position-self._destination)
        # Ohmni reach the destination
        if distance < 1:
            return True, 10
        # Compute current reward
        reward = 0
        if distance < 10:
            reward = 10 - distance
        else:
            reward = 0
        # Stop if detecting collision
        collision = self._env.getContactPoints()
        for contact in collision:
            # Contact with things different from floor
            if contact[2] != 0:
                return True, reward
        # Stop if Ohmni fall out of the environment
        if position[0] >= 10 or position[0] <= -10:
            return True, reward
        if position[1] >= 10 or position[1] <= -10:
            return True, reward
        if position[2] >= 0.5 or position[2] <= -0.5:
            return True, reward
        # Ohmni on his way
        return False, reward

    def _reset(self):
        """ Reset """
        self._num_steps = 0
        self._episode_ended = False
        self._env.reset()
        return ts.restart(np.zeros(self._image_dim, dtype=np.float32))

    def _step(self, action=4):
        """ Step, action is steering value with mean value 4 """
        self._num_steps += 1
        # If ended, reset the environment
        if self._episode_ended or self._num_steps > self._max_steps:
            return self.reset()
        # Step the environment
        left_wheel = THROTTLE_RANGE[1] + (action-4)/2
        right_wheel = THROTTLE_RANGE[1] - (action-4)/2
        self._env.step(left_wheel, right_wheel)
        # Compute and save states
        _, _, rgb_img, _, seg_img = self._env.capture_image()
        self._img = np.array(rgb_img, dtype=np.float32)/255
        mask = np.minimum(seg_img, 1, dtype=np.float32)
        self._state = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
        self._episode_ended, reward = self._compute_reward()
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
    def __init__(self, virtual=False):
        if virtual:
            pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
        self.name = 'OhmniInSpace-v0'

    def gen_env(self):
        """ Convert pyenv to tfenv """
        display = PyEnv()
        env = tf_py_environment.TFPyEnvironment(display)
        return env, display
