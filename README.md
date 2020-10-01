# Ohmni DRL Path Planning

## Introduction

In this project, we will explore a solution to apply deep reinforcement learning (DRL) to teach Ohmni avoiding obstacles and reaching its destination.
To leverage the cost of pricey devices such as lidars, depth cameras, etc., we plan to approach the problem by using an RGB camera. In the first place, we used a segementation network to draw floor and non-floor objects. Then using these knowledge, Ohmni will be trained by DRL to do a local path planning.
