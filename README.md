# Ohmni DRL Path Planning

## Introduction

In this project, we explores a solution to apply deep reinforcement (DRL) learning to teach Ohmni avoiding obstacles and reaching its ultimate goals.
To leverage the cost of pricey devices such as lidars, depth cameras, etc., we plan to approach the problem by using only one RGB camera. In the 
first place, we used a segementation network to draw floor and non-floor objects. Then using these knowledge, Ohmni will be trained by DRL to do a local 
path planning.
