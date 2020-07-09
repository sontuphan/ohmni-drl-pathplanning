import pybullet as p
import pybullet_data
import time
import cv2 as cv
import numpy as np

from env.objs import floor, ohmni, obstacle

clientId = p.connect(p.DIRECT)
p.setGravity(0, 0, -10, physicsClientId=clientId)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setRealTimeSimulation(1)

while True:
    time.sleep(1/60)
