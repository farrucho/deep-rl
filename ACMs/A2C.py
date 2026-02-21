from re import S

import gymnasium as gym
import os
from jax import checkpoint
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time
from env_wrappers import AtariEnv
from model import PolicyReinforceModel, PolicySimpleReinforceModel, ValueStateModel, ValueSimpleStateModel, PolicyAndStateModel
import wandb
import torch.multiprocessing as mp
from collections import deque
import gc

class A2C():
    def __init__(self, params, gym_env_name, num_workers, wand_name, checkpoint=None):
        self.params = params
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda")

        self.gym_env_name = gym_env_name
        self.envs = gym.vector.AsyncVectorEnv([lambda: AtariEnv(gym.make(self.gym_env_name), stack_size=4) for _ in range(0, num_workers)])

        self.observation_shape = self.envs.single_observation_space.shape
        self.num_actions = self.envs.single_action_space.n # type: ignore

        self.model = PolicyAndStateModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device)
        if checkpoint != None:
            self.model.load_state_dict(torch.load(checkpoint))
            print(f"model checkpoint loaded: {checkpoint}")
        
        self.num_workers = num_workers

        self.wandb_name = wand_name

        self._steps = 0
        self._episodes = 0

        self.deque_test_rewards = deque(maxlen=self.params["deque_test_rewards"])

        self.run = wandb.init(
            entity="franciscolaranjo9-personal",
            project="A2C",
            name=self.wandb_name,
            config=self.params
        )

        self.tq = tqdm()
        self.n_save = 1
        self.n_log = 0
        self.n_video = 1

    def generate_bootstrapping_episodes(self, last_observation, model):
        """
        execute n_step bootstraping
        note: if an episode ends the mask terminanted will handle it after, the following steps will be used to also train the model.
        """
        n_steps = self.params["n_step_bootstrapping"]

        ep_rewards = torch.zeros((self.num_workers, n_steps, 1)).to(self.device)
        ep_logprobs = torch.zeros((self.num_workers, n_steps, 1)).to(self.device)
        ep_state_values = torch.zeros((self.num_workers, n_steps, 1)).to(self.device)
        ep_entropies = torch.zeros((self.num_workers, n_steps, 1)).to(self.device)
        ep_bootstrap_values = torch.zeros((self.num_workers, 1)).to(self.device)
        ep_terminations = torch.zeros((self.num_workers, n_steps, 1)).to(self.device)

        for step in range(0, n_steps):
            self.tq.update(self.num_workers)
            last_observation_tensor = torch.from_numpy(last_observation).to(self.device).float()
            # print(last_observation_tensor.shape, "ver se esta a distribuir bem shape: (num_workers, 4, 84, 84)")

            model_output = model(last_observation_tensor)

            policy_actions = model_output[0]
            # print(policy_actions.shape, "shape: (num_workers, num_actions)")

            softmax_probs = nn.functional.softmax(policy_actions, dim=-1)
            # print(softmax_probs.shape, "ver se probabilidade estao certas e somam 1 nas linhas, shape: (num_workers, num_actions)")

            dist = torch.distributions.Categorical(softmax_probs)
            action = dist.sample() # retorna index
            # print(action.shape, "ver se esta a distribuir bem shape: (num_workers, )") # nao emter unsqueeze se nao logprob fica shape alterada

            logprob = dist.log_prob(action).unsqueeze(1)
            # print(logprob.shape, "ver se esta a distribuir bem shape: (num_workers, 1)")

            entropy = dist.entropy().unsqueeze(1)
            # print(entropy.shape, "ver se esta a distribuir bem shape: (num_workers, 1)")


            state_value = model_output[1]
            observations, rewards, terminations, truncations, infos = self.envs.step(action)

            terminations = np.array(terminations, dtype=np.int8) # true is 1
            
            last_observation = observations
            ep_rewards[:, step] = torch.from_numpy(rewards).unsqueeze(1).to(self.device).float()
            ep_logprobs[:, step] = logprob
            ep_state_values[:, step] = state_value
            ep_entropies[:, step] = entropy
            ep_terminations[:, step] = torch.from_numpy(terminations).unsqueeze(1).to(self.device).float()

            self._steps += self.num_workers
        self._episodes += self.num_workers

        ep_bootstrap_values = model(torch.from_numpy(last_observation).to(self.device).float())[1]
        
        return last_observation, ep_rewards, ep_logprobs, ep_state_values, ep_entropies, ep_terminations, ep_bootstrap_values


    def train(self):
        last_observation, info = self.envs.reset()


        while True:
            # generate episodes within n_step_bootstrapping and discounted_rewards
            last_observation, ep_rewards, ep_logprobs, ep_state_values, ep_entropies, ep_terminations, ep_bootstrap_values = self.generate_bootstrapping_episodes(last_observation, self.model)

            discounted_rewards = torch.zeros_like(ep_rewards, dtype=torch.float32, device=self.device)

            G = ep_bootstrap_values
            for t in reversed(range(self.params["n_step_bootstrapping"])):
                G = ep_rewards[:, t] + self.params["discount_factor"] * G * (1 - ep_terminations[:, t])
                discounted_rewards[:, t] = G


            # calculate loss
            advantages = discounted_rewards - ep_state_values # (num_workers, n_step_bootstrapping,1)
            
            actor_loss = - torch.mean(ep_logprobs*advantages.detach())

            critic_loss = nn.functional.mse_loss(ep_state_values, discounted_rewards.detach())
            
            entropy_loss = -ep_entropies.mean()

            loss = actor_loss + self.params["value_loss_coef"] * critic_loss + self.params["entropy_beta"] * entropy_loss

            self.model.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params["clip_grad"])

            self.model.optimizer.step()

            if self._steps - self.n_log*self.params["step_logging_k"] >= 0:
                self.run.log({
                    "loss/total": loss.item(),
                    "loss/actor": actor_loss.item(),
                    "loss/critic": self.params["value_loss_coef"] * critic_loss.item(),
                    "loss/entropy": self.params["entropy_beta"] * entropy_loss.item(),
                    "policy/entropy": ep_entropies.mean().item(),
                    "policy/mean_logprob": ep_logprobs.mean().item(),
                    "value/mean_value": ep_state_values.mean().item(),
                    "value/mean_return": discounted_rewards.mean().item(),
                    "value/mean_advantage": advantages.mean().item(),
                    "train/steps": self._steps,
                    "train/episodes": self._episodes,
                    "grad/global_norm": torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.params["clip_grad"]).item(),
                }, step=self._steps)
                self.n_log += 1
            if self._steps - self.n_video*self.params["test_video_k"] >= 0:
                test_env = AtariEnv(gym.make(self.gym_env_name), stack_size=4)
                obs, info = test_env.reset()
                terminated = False
                truncated = False
                frames = []
                rewards = []
                while not (terminated or truncated):
                    obs_tensor = torch.from_numpy(obs).to(self.device).float().unsqueeze(0)
                    with torch.no_grad():
                        model_output = self.model(obs_tensor)
                    policy_actions = model_output[0]
                    softmax_probs = nn.functional.softmax(policy_actions, dim=-1)

                    dist = torch.distributions.Categorical(softmax_probs)
                    action = dist.sample().item()

                    obs, reward, terminated, truncated, info = test_env.step(action)
                    rewards.append(reward)
                    frames.append(info["rgb_frame"])
                self.deque_test_rewards.append(np.sum(rewards))
                frames = np.stack(frames, 0).transpose(0, 3, 1, 2)
                self.run.log({"episode_video": wandb.Video(frames, caption=f"reward: {np.sum(rewards)}", fps=60, format="mp4"), "deque_test_rewards": np.mean(self.deque_test_rewards), "test_reward": np.sum(rewards)}, step=self._steps)
                self.n_video += 1
                
                # delete video memory, this fixes memory leaks
                del frames
                del rewards
                del ep_rewards
                del ep_logprobs
                del ep_state_values
                del ep_entropies
                del ep_terminations
                del ep_bootstrap_values
                del discounted_rewards
                del advantages
                del loss
                del actor_loss
                del critic_loss
                del entropy_loss
                test_env.close()
                gc.collect()
                torch.cuda.empty_cache()

            if self._steps - self.n_save*self.params["save_step_k"] >= 0:
                try:
                    os.makedirs(f"models/A2C/{self.gym_env_name}/{self.wandb_name}/model/")
                except:
                    print(f"saved checkpoint at step {self._steps}")
                    torch.save(self.model.state_dict(),f"models/A2C/{self.gym_env_name}/{self.wandb_name}/model/{self._steps}.pth")
                    self.n_save += 1
        
        


params = {
    "lr": 1e-4,
    "discount_factor": 0.99,
    "entropy_beta": 0.01,
    "n_step_bootstrapping": 64,
    "clip_grad": 40,
    "value_loss_coef": 10,


    # non related to algorithm
    "step_logging_k" : 1000,
    "test_video_k": 2000000,
    "deque_test_rewards" : 20, # TODO
    "save_step_k": 500000,
}

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    # torch.set_num_threads(1) # each process fits inside a cpu core whether physical or logical
    # torch.set_num_interop_threads(1)
    mp.set_start_method("spawn", force=True)# avoid memory duplication, instead of Fork+CUDA  
    a3c = A2C(params, gym_env_name="PongNoFrameskip-v4", num_workers=16, wand_name="(Part9)Pong 16 workers; less logging (also fixed)", checkpoint="models/A2C/PongNoFrameskip-v4/(Part8)Pong 16 workers; max steps per second/model/6400000.pth")
    
    a3c.train()
