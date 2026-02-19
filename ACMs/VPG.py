import gymnasium as gym
from jax import checkpoint
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time
from env_wrappers import AtariEnv
from model import PolicyReinforceModel, PolicySimpleReinforceModel, ValueStateModel
import wandb

def generate_episode(policy_model, state_values_model, env, max_steps=1000000):
    device = torch.device("cuda")
    
    last_observation, info = env.reset()
    rewards = []
    logprobs = []
    state_values = []
    entropies = []
    terminated, truncated = False, False
    _idx = 0
    frames = []

    while not (terminated or truncated) and _idx < max_steps:
        last_observation_tensor = torch.from_numpy(last_observation).unsqueeze(0).to(device, non_blocking=True).float()


        policy_actions = policy_model(last_observation_tensor)
        softmax_probs = nn.functional.softmax(policy_actions, dim=-1)
        dist = torch.distributions.Categorical(softmax_probs)
        action = dist.sample() # retorna index
        
        state_value = state_values_model(last_observation_tensor)
        logprob = dist.log_prob(action)
        entropy = dist.entropy() # this uses ln(6) - torch.sum(softmax_probs * torch.log2(softmax_probs), dim=-1) 


        observation, reward, terminated, truncated, info = env.step(action.item())


        last_observation = observation
        rewards.append(reward)
        logprobs.append(logprob)
        state_values.append(state_value)
        entropies.append(entropy)
        _idx += 1
        frames.append(info["rgb_frame"])

    return torch.tensor(rewards).to(device).unsqueeze(1), torch.stack(logprobs).to(device), torch.stack(state_values).to(device).squeeze(1), torch.stack(entropies).to(device), np.stack(frames, 0)



def main_vpg(gym_make_env, params, wand_name, policy_checkpoint=None, value_checkpoint=None):
    # --- wandb configuration ---
    run = wandb.init(
        entity="franciscolaranjo9-personal",
        project="VPG",
        name=wand_name,
        config = params
    )
    log_steps = 1 # logging every k steps (in this case episodes)
    # --------------------------

    device = torch.device("cuda")

    env = AtariEnv(gym.make(gym_make_env), stack_size=4)
    last_observation, info = env.reset()
    policy_model = PolicyReinforceModel(last_observation.shape, env.action_space.n, lr=params["lr"]).to(device)
    value_model = ValueStateModel(last_observation.shape, lr=params["lr"]).to(device)

    if policy_checkpoint is not None:
        policy_model.load_state_dict(torch.load(policy_checkpoint))
    if value_checkpoint is not None:
        value_model.load_state_dict(torch.load(value_checkpoint))


    last_episodes_length = 20
    save_steps = 50 # save model every k steps
    video_steps = 50 # get video every k steps

    try:
        last_episodes_rewards = deque(maxlen=last_episodes_length)
        
        _envsteps = 0
        
        tq = tqdm()
        while True:
            tq.update(1)

            rewards, logprobs, state_values, entropies, frames_of_episode = generate_episode(policy_model, value_model, env)

            discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)

            G = 0.0
            for j in reversed(range(len(rewards))):
                G = rewards[j] + G*params["discount_factor"]
                discounted_rewards[j] = G

            # calculate loss
            value_model_loss = torch.mean(torch.pow(discounted_rewards - state_values,2)) # do not normalize state values

            advantage_term_policy = discounted_rewards - state_values.detach()
            advantage_term_policy = (advantage_term_policy - advantage_term_policy.mean()) / (advantage_term_policy.std() + 1e-8) # normalize, in this case the mean will be 0
            policy_model_loss = -torch.mean((advantage_term_policy)*logprobs + params["beta"]*entropies) # detach aqui é essencial caso contrario a backprop vai para o value model, coisa que não queremos, pois so estamos a dar update ao step model

            value_model.optimizer.zero_grad() # zero the gradient buffers    
            value_model_loss.backward()
            value_model.optimizer.step()
            
            policy_model.optimizer.zero_grad() # zero the gradient buffers
            policy_model_loss.backward()
            policy_model.optimizer.step()

            


            _envsteps += 1

            last_episodes_rewards.append(torch.sum(rewards).item())
            if _envsteps % log_steps == 0:
                run.log({"policy_model_loss": policy_model_loss, "value_model_loss": value_model_loss, "episode_reward": torch.sum(rewards), "last_episodes_reward": np.mean(last_episodes_rewards), "mean_entropy": torch.mean(entropies).item(), "advantage_term_policy": torch.mean(advantage_term_policy).item()}, step=_envsteps)
            if _envsteps % save_steps == 0:
                torch.save(policy_model.state_dict(),f"models/{gym_make_env}/policy_model/{_envsteps}.pth")
                torch.save(value_model.state_dict(),f"models/{gym_make_env}/value_model/{_envsteps}.pth")
            if _envsteps % video_steps == 0:
                frames = frames_of_episode.transpose(0, 3, 1, 2)
                run.log({"episode_video": wandb.Video(frames, caption=f"{torch.sum(rewards)}", fps=30, format="mp4")}, step=_envsteps)



    except KeyboardInterrupt:
        print("interrompido")


params = {
    "lr": 1e-4,
    "discount_factor": 0.99,
    "beta": 1e-3 # for pong max entropy is -6*1/6*log2(1/6) = 2.585 or if dist.entropy() ln(6) = 1.79
}

main_vpg(gym_make_env="ALE/Pong-v5", params=params, wand_name="Pong - Part3", policy_checkpoint="models/ALE/Pong-v5/policy_model/1950.pth", value_checkpoint="models/ALE/Pong-v5/value_model/1950.pth")
