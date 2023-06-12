import gymnasium as gym
import numpy as np

if __name__ == '__main__':
    a = np.arange(9).reshape(3, 3)
    print(a)
    b = np.array([True, False, True])
    c = np.array([False, False, True])
    print(a[~b & ~c])
    envs = gym.vector.make("LunarLander-v2", num_envs=2)

    # envs.reset(seed=0)
    # for i in range(100):
    #     observations, rewards, termination, truncation, infos = envs.step(envs.action_space.sample())
    #     print(termination.dtype)
    #     if 'final_observation' in infos:
    #         print(infos['final_observation'])
    #         print(infos['_final_observation'])
    #         print(infos['final_info'])
    #         print(infos['_final_info'])
    #         print()
        
    # envs.close()