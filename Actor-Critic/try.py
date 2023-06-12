import gymnasium as gym
import numpy as np


envs = gym.vector.make("LunarLander-v2", num_envs=4)

envs.reset(seed=0)
for i in range(100):
    observations, rewards, termination, truncation, infos = envs.step(envs.action_space.sample())
    print(type(observations), observations.shape)
    if 'final_observation' in infos:
        print(infos['final_observation'])
        print(infos['_final_observation'])
        print(infos['final_info'])
        print(infos['_final_info'])
        print()
    
envs.close()