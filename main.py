import sys
import cv2 as cv

from env import test, OhmniInSpace
from src import train


if sys.argv[1] == '--test':
    if sys.argv[2] == 'debug':
        test.show()
    if sys.argv[2] == 'gui':
        oit = OhmniInSpace.PyEnv(gui=True)
    if sys.argv[2] == 'py-env':
        oit = OhmniInSpace.PyEnv(gui=False)
        while True:
            _, reward, _, observation = oit.step(action=4)
            oit.render()
            cv.imshow('Segmentation', observation)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    if sys.argv[2] == 'tf-env':
        oit = OhmniInSpace.TfEnv()
        tf_env, _ = oit.gen_env()
        print("TimeStep Specs:", tf_env.time_step_spec())
        print("Action Specs:", tf_env.action_spec())

elif sys.argv[1] == '--ohmni':
    if sys.argv[2] == 'train':
        train.train()

else:
    print("Error: Invalid option!")
