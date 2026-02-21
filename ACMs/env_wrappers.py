import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import FrameStackObservation
import ale_py
import time
import numpy as np
from collections import deque

class AtariEnv(gym.Wrapper):
    """
    This env was created to allow prepocessing the observation space and frame stacking features, according to the original paper, 
    """
    def __init__(self, env: Env, stack_size: int):
        super().__init__(env)
        self.env = env
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.observation_space = gym.spaces.Box(
            high=255,
            low=0,
            shape=(self.stack_size, 84, 84),
            dtype=np.uint8
        )

    def preprocess_observation(self, observation):
        # resize and crop to 84x84
        return cv2.resize(cv2.resize(cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY), (84, 110), interpolation=cv2.INTER_AREA)[18:101,:], (84,84))

    def save_stack(self, observation, filename="images/test.png"):
        frame_together = np.hstack(observation)
        frame_together = (frame_together).astype(np.uint8)
        cv2.imwrite(filename, frame_together)
        return frame_together

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        info["rgb_frame"] = obs

        self.frames.append(self.preprocess_observation(obs))
        
        # reward clipping
        return np.array(self.frames), np.sign(reward), terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info["rgb_frame"] = obs
        updated_observation = self.preprocess_observation(obs)

        for _ in range(0, self.stack_size):
            self.frames.append(updated_observation)
        return np.array(self.frames), info

    

        


# env = gym.make('ALE/Breakout-v5')
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
#         env.save_stack(obs,filename="test4.png")
#         print("saved")
#         time.sleep(1)
# except KeyboardInterrupt:
#     env.close()

