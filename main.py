import sys
import cv2 as cv
from env import test, OhmniInTemple


if sys.argv[1] == '--test':
    if sys.argv[2] == 'debug':
        test.show()
    if sys.argv[2] == 'gui':
        oit = OhmniInTemple.Environment(gui=True)
    if sys.argv[2] == 'env':
        oit = OhmniInTemple.Environment(gui=False)
        while True:
            img = oit.step(10, 10)
            cv.imshow('Segmentation', img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
elif sys.argv[1] == '--ohmni':
    pass

else:
    print("Error: Invalid option!")
