import pybullet as p
import pybullet_data
import numpy as np
import cv2 as cv

from env.objs import floor, ohmni, obstacle

THROTTLE_RANGE = [-15, 15]
STEERING_RANGE = [-4, 4]


class Simulator:
    def __init__(self, gui=False, num_of_obstacles=4, image_shape=(96, 96)):
        self.gui = gui
        self.timestep = 0.05
        self.num_of_obstacles = num_of_obstacles
        self.image_shape = image_shape
        self.clientId, self.get_velocities = self._init_ws()
        self.ohmniId, self.get_image = self._build()
        self.LEFT_WHEEL = 0
        self.RIGHT_WHEEL = 1

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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Set timestep and sliders corresponding to the mode
        throttle, steering = None, None
        if self.gui:
            p.setRealTimeSimulation(1)
            throttle = p.addUserDebugParameter(
                'Throttle', THROTTLE_RANGE[0], THROTTLE_RANGE[1], 0)
            steering = p.addUserDebugParameter(
                'Steering', STEERING_RANGE[0], STEERING_RANGE[1], 0)
        else:
            p.setTimeStep(self.timestep)
        # Utility

        def get_velocities():
            user_throttle = 0
            user_steering = 0
            if self.gui:
                user_throttle = p.readUserDebugParameter(throttle)
                user_steering = p.readUserDebugParameter(steering)
            return user_throttle, user_steering
        # Return
        return clientId, get_velocities

    def _build(self):
        """ Involving floor, ohmni, obstacles into the environment """
        # Add gravity
        p.setGravity(0, 0, -10, physicsClientId=self.clientId)
        # Add plane and ohmni
        floor(texture=False, wall=False)
        ohmniId, get_image = ohmni()
        # Add obstacles at random positions
        for _ in range(self.num_of_obstacles):
            obstacle()
        # Return
        return ohmniId, get_image

    def _start(self):
        """ This function is only called in gui mode """
        while True:
            _, _, _, _, seg_img = self.get_image(self.image_shape)
            throttle, steering = self.get_velocities()
            left_wheel, right_wheel = throttle+steering, throttle-steering
            self.step(left_wheel, right_wheel)
            mask = np.minimum(seg_img, 1, dtype=np.float)
            cv.imshow('Segmentation', mask)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    def _reset(self):
        """ Remove all objects, then rebuild them """
        p.resetSimulation()
        self._build()

    def reset(self):
        """ Reset the environment """
        self._reset()

    def step(self, left_wheel, right_wheel):
        """ Controllers for left/right wheels which are separate """
        p.setJointMotorControl2(self.ohmniId, self.LEFT_WHEEL,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel)
        p.setJointMotorControl2(self.ohmniId, self.RIGHT_WHEEL,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel)


class Environment:
    def __init__(self, gui=False, image_shape=(96, 96)):
        self.throttle_range = THROTTLE_RANGE
        self.steering_range = STEERING_RANGE
        self.num_of_obstacles = 5
        self.image_shape = image_shape
        # Init bullet server
        self.s = Simulator(
            gui,
            num_of_obstacles=self.num_of_obstacles,
            image_shape=self.image_shape
        )
        self.destination = np.array([10, 0, 0], dtype=np.float)

    def _compute_reward(self):
        """ Compute reward """
        collision = p.getContactPoints(self.s.ohmniId)
        for contact in collision:
            if contact[2] != 0:  # contact with thing different to floor
                return -1
        position, _ = p.getBasePositionAndOrientation(self.s.ohmniId)
        position = np.array(position, dtype=np.float)
        # Ohmni fall out of the environment
        if position[0] >= 10 or position[0] <= -10:
            return -1
        if position[1] >= 10 or position[1] <= -10:
            return -1
        if position[2] >= 0.5 or position[2] <= -0.5:
            return -1
        # Ohmni reach the destination
        state = np.linalg.norm(position-self.destination)
        return 1 if state < 1 else 0

    def step(self, left_wheel=0, right_wheel=0):
        """ Step """
        self.s.step(left_wheel, right_wheel)
        _, _, _, _, seg_img = self.s.get_image(self.image_shape)
        p.stepSimulation()
        mask = np.minimum(seg_img, 1, dtype=np.float)
        reward = self._compute_reward()
        if reward != 0:
            print(reward)
            self.s.reset()
        return mask
