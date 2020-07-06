import pybullet as p


def plane():
    return p.loadURDF('plane.urdf')


def floor():
    return p.loadURDF('env/model/floor.urdf')


def car():
    return p.loadURDF('env/model/car.urdf', [0, 0, 0.1])


def ohmni():
    return p.loadURDF('env/model/ohmni.urdf', [0, 0, 0.1])
