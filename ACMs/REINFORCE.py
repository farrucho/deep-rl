import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time
from env_wrappers import AtariEnv
from model import PolicyReinforceModel, PolicySimpleReinforceModel
import wandb

def generate_episode(policy, env, max_steps=1000000):
    device = torch.device("cuda")
    
    last_observation, info = env.reset()
    rewards = []
    logprobs = []
    terminated, truncated = False, False
    _idx = 0
    frames = []

    while not (terminated or truncated) and _idx < max_steps:
        last_observation_tensor = torch.from_numpy(last_observation).unsqueeze(0).to(device, non_blocking=True).float()


        policy_actions = policy(last_observation_tensor)
        softmax_probs = nn.functional.softmax(policy_actions, dim=-1)
        dist = torch.distributions.Categorical(softmax_probs)
        action = dist.sample() # retorna index
        
        observation, reward, terminated, truncated, info = env.step(action.item())



        last_observation = observation
        rewards.append(reward)
        logprobs.append(dist.log_prob(action))
        _idx += 1
        frames.append(info["rgb_frame"])

    return torch.tensor(rewards, dtype=torch.float32).to(device), torch.stack(logprobs).to(device), np.stack(frames, 0)



def main_reinforce(gym_make_env, params, wand_name, checkpoint=None, atariEnv=True):
    # --- wandb configuration ---
    run = wandb.init(
        entity="franciscolaranjo9-personal",
        project="REINFORCE",
        name=wand_name,
        config = params
    )
    log_steps = 1 # logging every k steps (in this case episodes)
    # --------------------------

    device = torch.device("cuda")
    if atariEnv:
        env = AtariEnv(gym.make(gym_make_env), stack_size=4)
        last_observation, info = env.reset()
        policy = PolicyReinforceModel(last_observation.shape, env.action_space.n, lr=params["lr"]).to(device)
    else:
        env = gym.make(gym_make_env)
        last_observation, info = env.reset()
        policy = PolicySimpleReinforceModel(last_observation.shape, env.action_space.n, lr=params["lr"]).to(device)

    
    if checkpoint is not None:
        policy.load_state_dict(torch.load(checkpoint))


    last_episodes_length = 20
    save_steps = 50 # save model every k steps
    video_steps = 50 # get video every k steps

    try:
        last_episodes_rewards = deque(maxlen=last_episodes_length)
        
        _envsteps = 0
        
        tq = tqdm()
        while True:
            tq.update(1)

            # generate episode following the policy get G(tau)
            rewards, logprobs, frames_of_episode = generate_episode(policy, env)

            # calculate loss
            discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8) # normalize

            G = 0.0
            for j in reversed(range(len(rewards))):
                G = rewards[j] + G*params["discount_factor"]
                discounted_rewards[j] = G


            loss = -torch.sum(logprobs*discounted_rewards).to(device)

            # policy training step
            policy.optimizer.zero_grad() # zero the gradient buffers
            loss.backward()
            policy.optimizer.step()


            _envsteps += 1

            last_episodes_rewards.append(torch.sum(rewards).item())
            if _envsteps % log_steps == 0:
                run.log({"loss": loss, "episode_reward": torch.sum(rewards), "last_episodes_reward": np.mean(last_episodes_rewards)}, step=_envsteps)
            if _envsteps % save_steps == 0:
                torch.save(policy.state_dict(),f"models/{gym_make_env}/{_envsteps}.pth")
            if _envsteps % video_steps == 0:
                frames = frames_of_episode.transpose(0, 3, 1, 2)
                run.log({"episode_video": wandb.Video(frames, caption=f"{torch.sum(rewards)}", fps=30, format="mp4")}, step=_envsteps)



    except KeyboardInterrupt:
        print("interrompido")


params = {
    "lr": 1e-4,
    "discount_factor": 0.99
}

main_reinforce(gym_make_env="ALE/Pong-v5", params=params, wand_name="REINFORCE Pong Normalized Discounted Rewards")
