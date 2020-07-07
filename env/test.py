import pybullet as p
import pybullet_data

from env.objs import floor, ohmni

VELOCITY = 30


def init_ws():
    clientId = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=clientId)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    return clientId


def show():
    init_ws()
    angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
    throttle = p.addUserDebugParameter('Throttle', 0, 1, 0)
    floor(texture=True, wall=True)
    ohmniId, get_image = ohmni()

    while True:
        get_image((224, 224))
        user_angle = p.readUserDebugParameter(angle)
        user_throttle = p.readUserDebugParameter(throttle)
        left_wheel = VELOCITY*(user_throttle+user_angle)
        right_wheel = VELOCITY*(user_throttle-user_angle)
        print(left_wheel, right_wheel)
        p.setJointMotorControl2(ohmniId, 0,
                                p.VELOCITY_CONTROL,
                                targetVelocity=left_wheel)
        p.setJointMotorControl2(ohmniId, 1,
                                p.VELOCITY_CONTROL,
                                targetVelocity=right_wheel)
        p.stepSimulation()
