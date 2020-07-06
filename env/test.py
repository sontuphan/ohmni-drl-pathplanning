import pybullet as p
import pybullet_data

from env.objs import floor, ohmni, plane


def init_ws():
    clientId = p.connect(p.GUI)
    p.setGravity(0, 0, -10, physicsClientId=clientId)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print(pybullet_data.getDataPath())
    return clientId


def show():
    clientId = init_ws()
    angle = p.addUserDebugParameter('Steering', -0.5, 0.5, 0)
    throttle = p.addUserDebugParameter('Throttle', 0, 20, 0)
    planeId = floor()
    # carId = car()
    ohmniId = ohmni()

    # wheel_indices = [1, 3, 4, 5]
    # hinge_indices = [0, 2]
    # while True:
    #     user_angle = p.readUserDebugParameter(angle)
    #     user_throttle = p.readUserDebugParameter(throttle)
    #     for joint_index in wheel_indices:
    #         p.setJointMotorControl2(carId, joint_index,
    #                                 p.VELOCITY_CONTROL,
    #                                 targetVelocity=user_throttle)
    #     for joint_index in hinge_indices:
    #         p.setJointMotorControl2(carId, joint_index,
    #                                 p.POSITION_CONTROL,
    #                                 targetPosition=user_angle)
    #     p.stepSimulation()
    while True:
        p.stepSimulation()
