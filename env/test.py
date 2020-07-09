import pybullet as p
import pybullet_data
import time
import cv2 as cv
import numpy as np

from env.objs import floor, ohmni, obstacle

VELOCITY = 30


def init_ws():
    clientId = p.connect(p.DIRECT)
    p.setGravity(0, 0, -10, physicsClientId=clientId)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return clientId


def show():
    init_ws()
    # p.setRealTimeSimulation(1)
    angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
    throttle = p.addUserDebugParameter('Throttle', -0.5, 0.5, 0)
    floor(texture=True, wall=True)
    ohmniId, get_image = ohmni()
    obstacle()
    obstacle()
    obstacle()
    obstacle()

    while True:
        start = time.time()

        width, height, rgb_img, depth_img, seg_img = get_image((96, 96))
        user_angle = p.readUserDebugParameter(angle)
        user_throttle = p.readUserDebugParameter(throttle)
        left_wheel = VELOCITY*(user_throttle+user_angle)
        right_wheel = VELOCITY*(user_throttle-user_angle)
        p.setJointMotorControl2(ohmniId, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel)
        p.setJointMotorControl2(ohmniId, 1,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel)
        p.stepSimulation()
        mask = np.minimum(seg_img, 1, dtype=np.float)
        cv.imshow('Segmentation', mask)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

        end = time.time()
        print('Total estimated time: {:.4f}'.format(end-start))
