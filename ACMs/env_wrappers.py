import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import FrameStackObservation
import ale_py
import time
import numpy as np

class AtariEnv(FrameStackObservation):
    """
    This env was created to allow prepocessing the observation space and frame stacking features, according to the original paper, 
    """
    def __init__(self, env: Env, stack_size: int):
        super().__init__(env, stack_size)
        self.env = env
        self.n = stack_size
        self.previous_obs = np.zeros((self.n, 84,84))

    def preprocess_observation(self, observation):
        # resize to 84x84
        new_obs = np.zeros((self.n, 84,84))
        for j in range(0,4):
            new_obs[j] = cv2.resize(cv2.resize(cv2.cvtColor(observation[j], cv2.COLOR_RGB2GRAY), (84, 110), interpolation=cv2.INTER_AREA)[18:101,:], (84,84))
        return new_obs

    def save_stack(self, observation, filename="images/test.png"):
        frame_together = np.hstack(observation)
        frame_together = (frame_together * 255).astype(np.uint8)
        cv2.imwrite(filename, frame_together)
        return frame_together

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        info["rgb_frame"] = obs[-1]

        updated_observation = np.zeros((self.n, 84,84))
        for j in range(0,self.n-1):
            updated_observation[j] = self.previous_obs[j+1]
        
        # PONG
        updated_observation[-1] = cv2.resize(cv2.resize(cv2.cvtColor(obs[-1], cv2.COLOR_RGB2GRAY), (84, 110), interpolation=cv2.INTER_AREA)[18:101,:], (84,84)) # este é o mais recente
        
        
        self.previous_obs = updated_observation

        # reward clipping
        return updated_observation, np.sign(reward), terminated, truncated, info
    
    def reset(self):
        obs, info = super().reset()
        info["rgb_frame"] = obs[-1]
        updated_observation = self.preprocess_observation(obs)

        self.previous_obs = updated_observation
        return updated_observation, info

    

        


# # env = gym.make('ALE/Breakout-v5')
# env = gym.make('ALE/Pong-v5')
# env = AtariEnv(env, stack_size=4)
# print(env.action_space)
# observation, info = env.reset()
# i = 0
# try:
#     while i < 100:
#         # action = int(input())
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.save_stack(obs*255,filename="test4.png")
#         print("saved")
#         time.sleep(1)
# except KeyboardInterrupt:
#     env.close()

