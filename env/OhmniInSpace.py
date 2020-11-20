import pybullet as p
import pybullet_data
import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2 as cv

from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from env.objs import floor, ohmni, obstacle

VELOCITY_COEFFICIENT = 15
INTERPRETER = [[-0.4, -0.4], [-0.15, 0.15],
               [0., 0.], [0.15, -0.15], [0.4, 0.4]]


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
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # Return
        return clientId

    def _randomize_destination(self):
        # destination = (np.random.rand(2)*20-10).astype(dtype=np.float32)
        destination = np.array([8, 8], dtype=np.float32)
        p.addUserDebugLine(
            np.append(destination, 0.),  # From
            np.append(destination, 3.),  # To
            [1, 0, 0],  # Red
            physicsClientId=self.clientId
        )
        return destination

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
        self.destination = self._randomize_destination()

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
        self.image_stack = self.image_shape + (3,)
        self._num_of_obstacles = 0
        self._max_steps = 500
        # PyEnvironment variables
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,  minimum=0, maximum=4, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=self.image_stack, dtype=np.float32,
            minimum=0, maximum=1, name='observation')
        # Init bullet server
        self._env = Env(
            gui,
            num_of_obstacles=self._num_of_obstacles,
            image_shape=self.image_shape
        )
        # Internal states
        self._num_steps = 0
        self._episode_ended = False
        self._img = None
        self._state = None
        self._compute_state()

    def _get_image_state(self):
        _, _, rgb_img, _, seg_img = self._env.capture_image()
        img = np.array(rgb_img, dtype=np.float32)/255
        mask = np.minimum(seg_img, 1, dtype=np.float32)
        return img, mask

    def _get_pose_state(self):
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        destination_posistion = np.append(self._env.destination, 0.)
        rotation = R.from_quat(
            [-orientation[0], -orientation[1], -orientation[2], orientation[3]])
        rel_position = rotation.apply(destination_posistion - position)
        _pose = rel_position[0:2]
        return _pose.astype(dtype=np.float32)

    def _normalized_distance_to_destination(self):
        """ Compute the distance from agent to destination """
        position, _ = self._env.getBasePositionAndOrientation()
        position = np.array(position[0:2], dtype=np.float32)
        distance = np.linalg.norm(position-self._env.destination)
        origin = np.linalg.norm(np.zeros([2])-self._env.destination)
        return min(distance/origin, 1)

    def _is_fatal(self):
        """ Compute whether there are collisions or not """
        # If exceed the limitation of steps, return rewards
        if self._num_steps > self._max_steps:
            return True
        position, orientation = self._env.getBasePositionAndOrientation()
        position = np.array(position, dtype=np.float32)
        collision = self._env.getContactPoints()
        for contact in collision:
            # Contact with things different from floor
            if contact[2] != 0:
                return True
        # Ohmni felt out of the environment
        if abs(position[2]) >= 0.5:
            return True
        # Ohmni is falling down
        if abs(orientation[0]) > 0.2 or abs(orientation[1]) > 0.2:
            return True
        return False

    def _compute_reward(self):
        """ Compute reward and return (<stopped>, <reward>) """
        pose = self._get_pose_state()
        heading = np.array([1, 0])
        cosine_sim = np.inner(pose, heading) / \
            (np.linalg.norm(pose)*np.linalg.norm(heading))
        # Ohmni reaches the destination
        normalized_distance = self._normalized_distance_to_destination()
        shaped_reward = -normalized_distance
        if normalized_distance < 0.1:
            # The reward should be defined based on the discount
            return True, 1
        # Stop if detecting collisions or a fall
        if self._is_fatal():
            return True, -1
        # Ohmni on his way
        return False, shaped_reward

    def _reset(self):
        """ Reset environment"""
        self._env.reset()
        self._num_steps = 0
        self._episode_ended = False
        self._img = None
        self._state = None
        self._compute_state()
        return ts.restart(self._state)

    def _compute_state(self):
        if self._state is None:
            self._state = np.zeros(self.image_stack, dtype=np.float32)
        _img, _mask = self._get_image_state()  # Image state
        _pose = self._get_pose_state()  # Pose state
        # Gamifying
        (h, w) = self.image_shape
        _cent = np.array([w/2, h/2], dtype=np.float32)
        _dest = -_pose*1000 + _cent  # Transpose/Scale/Tranform
        normalized_distance = self._normalized_distance_to_destination()
        _color = normalized_distance - 0.05
        _thickness = 5  # Max conv kenel size
        _mask = cv.line(_mask,
                        (int(_cent[1]), int(_cent[0])),
                        (int(_dest[1]), int(_dest[0])),
                        _color, thickness=_thickness)
        _mask = _mask[..., np.newaxis]
        self._img = _img
        self._state = self._state[:, :, 1:]
        self._state = np.append(self._state, _mask, axis=2)

    def _step(self, action):
        """ Step, action is velocities of left/right wheel """
        # Reset if ended
        if self._episode_ended:
            return self.reset()
        self._num_steps += 1
        # Step the environment
        self._env.step(action)
        # Compute and save states
        self._compute_state()
        self._episode_ended, reward = self._compute_reward()
        # Transition
        if self._episode_ended:
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward)

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
