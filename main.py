import sys
import cv2 as cv

from env import OhmniInSpace
from src import train


if sys.argv[1] == '--test':
    if sys.argv[2] == 'py-env':
        ois = OhmniInSpace.PyEnv(gui=True)
        timestep = ois.reset()
        counter = 0
        while True:
            if timestep.is_last():
                counter = 0
            counter += 1
            timestep = ois.step(action=4)
            (_, reward, discount, observation) = timestep
            print('Step {} / Reward {}'.format(counter, reward))
            ois.render()
            cv.imshow('Segmentation', observation)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
    if sys.argv[2] == 'tf-env':
        ois = OhmniInSpace.TfEnv()
        tf_env = ois.gen_env()
        print("TimeStep Specs:", tf_env.time_step_spec())
        print("Action Specs:", tf_env.action_spec())

elif sys.argv[1] == '--ohmni':
    if sys.argv[2] == 'train':
        train.train()
    if sys.argv[2] == 'run':
        train.run()

else:
    print("Error: Invalid option!")
