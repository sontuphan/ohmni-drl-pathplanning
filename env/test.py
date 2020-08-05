import pybullet as p
import pybullet_data
import time
import cv2 as cv
import numpy as np

from env.objs import floor, ohmni, obstacle

VELOCITY = 15


def init_ws(gui=False, timestep=None):
    clientId = p.connect(p.GUI if gui else p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=clientId)
    if timestep is not None:
        p.setTimeStep(timestep)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if gui:
        throttle = p.addUserDebugParameter('Throttle', -1, 1, 0)
        angle = p.addUserDebugParameter('Angle', -0.25, 0.25, 0)

    def get_velocities():
        user_angle = 0
        user_throttle = 0
        if gui:
            user_throttle = p.readUserDebugParameter(throttle)
            user_angle = p.readUserDebugParameter(angle)
        return user_throttle, user_angle

    return clientId, get_velocities


def show():
    # Init bullet server
    clientId, get_velocities = init_ws(gui=True, timestep=0.05)
    # Add ground and ohmni
    floor(clientId, texture=True, wall=True)
    ohmniId, get_image = ohmni(clientId)
    # Add obstacles at random positions
    for i in range(4):
        obstacle(clientId)

    while True:
        start = time.time()

        width, height, rgb_img, depth_img, seg_img = get_image((96, 96))
        user_throttle, user_angle = get_velocities()
        left_wheel = VELOCITY*(user_throttle+user_angle)
        right_wheel = VELOCITY*(user_throttle-user_angle)
        p.setJointMotorControl2(ohmniId, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel)
        p.setJointMotorControl2(ohmniId, 1,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel)
        p.stepSimulation()
        position, _ = p.getBasePositionAndOrientation(ohmniId)
        position = np.array(position, dtype=np.float)
        destination = np.array([10, 0, 0], dtype=np.float)
        state = np.linalg.norm(destination-position)
        if state < 1:
            break
        mask = np.minimum(seg_img, 1, dtype=np.float)
        cv.imshow('Segmentation', mask)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        end = time.time()
        print('Total estimated time: {:.4f}'.format(end-start))
