import cv2
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import FrameStackObservation
import ale_py
import time
import numpy as np

class BreakoutEnv(FrameStackObservation):
    """
    This env was created to allow prepocessing the observation space and frame stacking features, according to the original paper, 
    """
    def __init__(self, env: Env, stack_size: int):
        super().__init__(env, stack_size)
        self.env = env
        self.n = stack_size

    def preprocess_observation(self, observation):
        # resize to 84x84 like in the paper
        new_obs = np.zeros((self.n, 84,84))
        for j in range(0,4):
            new_obs[j] = cv2.resize(cv2.cvtColor(observation[j], cv2.COLOR_RGB2GRAY), (84,84))
        return new_obs/255
        # frame_together = np.hstack(new_obs)
        # return frame_together

    def save_stack(self, observation, filename="images/test.png"):
        frame_together = np.hstack(observation)
        frame_together = (frame_together * 255).astype(np.uint8)
        cv2.imwrite(filename, frame_together)
        return frame_together

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["rgb_frame"] = obs[-1]
        updated_observation = self.preprocess_observation(obs)
        
        # reward clipping, o objetivo é destruir todos os blocos nao ha limite de tempo
        return updated_observation, np.sign(reward), terminated, truncated, info
    
    def reset(self):
        obs, info = super().reset()
        info["rgb_frame"] = obs[-1]
        updated_observation = self.preprocess_observation(obs)

        return updated_observation, info

    

        


# env = gym.make('ALE/Breakout-v5')
# env = BreakoutEnv(env, stack_size=4)

# observation, info = env.reset()
# i = 0
# try:
#     while i < 100:
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.save_stack(obs)
#         print("saved")
#         time.sleep(1)
# except KeyboardInterrupt:
#     env.close()

