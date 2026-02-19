import gymnasium as gym
from sympy import false
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time
from env_wrappers import AtariEnv
from model import PolicyReinforceModel, PolicySimpleReinforceModel, ValueStateModel, ValueSimpleStateModel
import wandb
import torch.multiprocessing as mp



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
    exit_loop = False

    while not truncated and not exit_loop:
        last_observation_tensor = torch.from_numpy(last_observation).unsqueeze(0).float()


        policy_actions = policy_model(last_observation_tensor)
        softmax_probs = nn.functional.softmax(policy_actions, dim=-1)
        dist = torch.distributions.Categorical(softmax_probs)
        action = dist.sample() # retorna index
        
        state_value = state_values_model(last_observation_tensor)
        logprob = dist.log_prob(action)
        entropy = dist.entropy() # this uses ln(6) - torch.sum(softmax_probs * torch.log2(softmax_probs), dim=-1) 


        observation, reward, terminated, truncated, info = env.step(action.item())

        if terminated:
            reward = 0
            exit_loop = True
        if _idx == max_steps:
            reward = state_values_model(torch.from_numpy(observation).unsqueeze(0).to(device, non_blocking=True).float())
            exit_loop = True

        last_observation = observation
        rewards.append(reward)
        logprobs.append(logprob)
        state_values.append(state_value)
        entropies.append(entropy)
        # frames.append(info["rgb_frame"])
        frames.append(0)
        _idx += 1



    return torch.tensor(rewards).to(device).unsqueeze(1), torch.stack(logprobs).to(device), torch.stack(state_values).to(device).squeeze(1), torch.stack(entropies).to(device), np.stack(frames, 0)





def work(rank, shared_policy_model, shared_value_model, n_step_bootstrapping, run):
    device = torch.device("cuda")
    # change params to respect params variable
    
    print(f"Worker {rank}")

    env = gym.make("CartPole-v1")

    _episode = 0
    while True:
        last_observation, info = env.reset()


        local_policy_model = PolicySimpleReinforceModel(last_observation.shape, env.action_space.n, lr=1e-4)
        local_value_model = ValueSimpleStateModel(last_observation.shape, lr=1e-4)

        local_policy_model.load_state_dict(shared_policy_model.state_dict())
        local_value_model.load_state_dict(shared_value_model.state_dict())


        # ---generate episode ---
        rewards, logprobs, state_values, entropies, frames_of_episode = generate_episode(local_policy_model, local_value_model, env, max_steps=n_step_bootstrapping)

        # optimize model
        discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=device)
        G = 0.0
        for j in reversed(range(0,len(rewards))):
            G = rewards[j] + G*params["discount_factor"]
            discounted_rewards[j] = G

        # --- calculate loss---
        local_value_model_loss = torch.mean(torch.pow(discounted_rewards - state_values,2)) # do not normalize state values

        advantage_term_policy = discounted_rewards - state_values.detach()
        advantage_term_policy = (advantage_term_policy - advantage_term_policy.mean()) / (advantage_term_policy.std() + 1e-8) # normalize, in this case the mean will be 0
        local_policy_model_loss = -torch.mean((advantage_term_policy)*logprobs + params["beta"]*entropies) # detach aqui é essencial caso contrario a backprop vai para o value model, coisa que não queremos, pois so estamos a dar update ao step model


        # --- optimize --- do i need lock here? # TODO
        shared_policy_model.zero_grad()
        local_policy_model_loss.backward()
        # TODO clip gradientes? nn.utils.clip_grad_norm_(local_policy_model.parameters(), policy_max_grad_norm)

        # copy gradients from the local model to shared model
        for param, shared_param in zip(local_policy_model.parameters(), shared_policy_model.parameters()):
            if shared_param.grad is None:
                    shared_param._grad = param.grad
            else:
                print("DEU TRIGGER NONE É SUPOSTO???")

        shared_policy_model.optimizer.step()
        local_policy_model.load_state_dict(shared_policy_model.state_dict())


        shared_value_model.zero_grad()
        local_value_model_loss.backward()
        # TODO clip gradientes? nn.utils.clip_grad_norm_(local_policy_model.parameters(), policy_max_grad_norm)

        # copy gradients from the local model to shared model
        for param, shared_param in zip(local_value_model.parameters(), shared_value_model.parameters()):
            if shared_param.grad is None:
                    shared_param._grad = param.grad
            else:
                print("DEU TRIGGER NONE É SUPOSTO???")

        shared_value_model.optimizer.step()
        local_value_model.load_state_dict(shared_value_model.state_dict())

        _episode += 1
        print(f"Worker {rank} | episode {_episode} | reward {rewards.sum().item()} | ")
        




    
def main_A3C(gym_make_env, params, wand_name):
    # --- wandb configuration ---
    run = wandb.init(
        entity="franciscolaranjo9-personal",
        project="A3C",
        name=wand_name,
        config = params
    )
    log_steps = 1 # logging every k steps (in this case episodes)
    # --------------------------

    env = gym.make(gym_make_env)
    last_observation, info = env.reset()

    global_policy_model = PolicySimpleReinforceModel(last_observation.shape, env.action_space.n, lr=1e-4)
    global_value_model = ValueSimpleStateModel(last_observation.shape, lr=1e-4)

    global_policy_model.share_memory()
    global_value_model.share_memory()

    num_workers = 3
    processes = []
    for j in range(0, num_workers):
        p = mp.Process(target=work, args=(j, global_policy_model, global_value_model, params["n_step_bootstrapping"], run))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("olaa2")


params = {
    "lr": 1e-4,
    "n_step_bootstrapping": 1000,
    "discount_factor": 0.99,
    "beta": 1e-3 # for pong max entropy is -6*1/6*log2(1/6) = 2.585 or if dist.entropy() ln(6) = 1.79
}

main_A3C(gym_make_env="CartPole-v1", params=params, "CartPole First Try")
print("olaa")

# FAZER CLASS  nao vale a pena adicionar compleixdfa a isto se depois vou converter em classe, converte em classe confia





