import pybullet as p
import numpy as np


def plane():
    return p.loadURDF('plane.urdf')


def floor(texture=False, wall=False):
    floorId = p.loadURDF('env/model/floor.urdf')
    if texture:
        textureId = p.loadTexture('env/model/texture/xwood1.jpg')
        p.changeVisualShape(floorId, -1, textureUniqueId=textureId)
    if wall:
        p.loadURDF('samurai.urdf')
    return floorId


def car():
    return p.loadURDF('env/model/car.urdf', [0, 0, 0.1])


def ohmni():
    start_pos = [0, 0, 0.1]
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    ohmniId = p.loadURDF('env/model/ohmni.urdf', start_pos, start_orientation)
    # ohmniId = p.loadURDF('env/model/simplecar.urdf', start_pos, start_orientation)

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=80.0, aspect=1.0, nearVal=0.01, farVal=50)

    def get_nav_image(img_shape=(96, 96)):
        # Center of mass position and orientation (of tube)
        tube_position, tube_orientation, _, _, _, _ = p.getLinkState(
            ohmniId, 4, computeForwardKinematics=True)
        tube_position = np.array(tube_position, dtype=np.float)
        rotation = np.array(p.getMatrixFromQuaternion(
            tube_orientation), dtype=np.float).reshape(3, 3)
        # Initial vectors
        init_eye_pos = np.array([0.015, 0, 0.4], dtype=np.float)
        init_target_pos = np.array([0.1, 0, -1], dtype=np.float)
        init_up_vector = np.array([0, 0, 1], dtype=np.float)
        # Rotated vectors
        eye_pos = tube_position + np.dot(rotation, init_eye_pos)
        target_pos = eye_pos + np.dot(rotation, init_target_pos)
        up_vector = np.dot(rotation, init_up_vector)
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye_pos,
            cameraTargetPosition=target_pos,
            cameraUpVector=up_vector)
        width, height = img_shape
        return p.getCameraImage(width, height, view_matrix, projection_matrix)

    return ohmniId, get_nav_image
