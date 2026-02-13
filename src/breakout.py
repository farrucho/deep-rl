from random import sample
from replaybuffer import ReplayBuffer, Sars
from model import Model, ConvModel
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time
from env_wrappers import BreakoutEnv


def run_test_episode(model, env, max_steps=1000):
    device = torch.device("cuda")
    last_observation, info = env.reset()

    frames = []
    total_reward = 0
    idx = 0
    terminated, truncated = False, False
    while not (terminated or truncated) and idx < max_steps:
        last_observation_tensor = torch.from_numpy(last_observation).unsqueeze(0).to(device, non_blocking=True).float()
        with torch.no_grad():
            best_q_action = model(last_observation_tensor).detach().max(-1).indices.item()
        observation, reward, terminated, truncated, info = env.step(best_q_action)
        last_observation = observation
        total_reward += reward
        idx += 1
        frames.append(info["rgb_frame"])


    return total_reward, np.stack(frames, 0)


def train_step(model, replay_buffer, target_model, discount_factor=0.99, use_huber_loss=False, use_DDQN=True):
    device = torch.device("cuda")

    current_states, actions, rewards, next_states, terminated, truncated = replay_buffer.sample()
    # current_states = torch.stack([torch.Tensor(e.state) for e in experiences]).to(device)
    # actions = torch.tensor([e.action for e in experiences]).to(device).unsqueeze(1) # shape (N,1); unsqueeze:"Returns a new tensor with a dimension of size one inserted at the specified position."
    # rewards = torch.tensor([torch.tensor(e.reward) for e in experiences]).to(device)
    # next_states = torch.stack([torch.Tensor(e.next_state) for e in experiences]).to(device)
    # terminated = torch.stack([torch.Tensor([0]) if e.terminated else torch.Tensor([1]) for e in experiences]).to(device)
    # truncated = torch.stack([torch.Tensor([0]) if e.truncated else torch.Tensor([1]) for e in experiences]).to(device)

    model_q_values = model(current_states) # [Q(s,a_0; \theta_{i}), Q(s,a_1; \theta_{i}), ... , Q(s,a_num_actions; \theta_{i})] shape: (N, num_actions)
    # nos nao queremos isto, queremos ir buscar especificamente Q(s,a; \theta_{i}), temos que usar o vetor actions
    model_q_a_values = torch.gather(model_q_values, 1, actions) # isto funciona porque actions é 0 ou 1, caso contrário seria preciso one hot encoding, para cada linha do model_q_values vai buscar a coluna action (0 ou 1) respetiva, obtendo corretamente Q(s,a; \theta_{i})
    # https://docs.pytorch.org/docs/stable/generated/torch.gather.html

    if not use_DDQN:
        with torch.no_grad():
            target_q_values_nextstate = target_model(next_states).to(device) # Q(s',a'; \theta^{-}) shape: (N, num_actions)
        max_q_target = torch.max(target_q_values_nextstate, 1, keepdim=True).values # (N,1) max values over each line (dimension to reduce is over the number of actions)
        
    else:
        with torch.no_grad():
            online_q_values_nextstate = model(next_states).to(device) # Q(s',a'; \theta^{i}) shape: (N, num_actions)
            target_q_values_nextstate = target_model(next_states).to(device)
        
        max_q_online_actions = torch.max(online_q_values_nextstate, 1).indices.unsqueeze(1) # shape: (N, 1)
        
        max_q_target = target_q_values_nextstate.gather(1, max_q_online_actions)

        # One hot para caso futuro:
        # F = nn.functional.one_hot
        # one_hot_best_max_q_online_actions = F(max_q_online_actions, num_classes=4) # (64, 4)
        # max_q_target = (target_q_values_nextstate * one_hot_best_max_q_online_actions).sum(dim=1, keepdim=True) # (64, 1)


    # falta discount factor
    if not use_huber_loss:
        loss = torch.mean((rewards.unsqueeze(1) + discount_factor*(1-terminated).unsqueeze(1)*max_q_target - model_q_a_values)**2) # eq. in pag. 260 Será que vai dar erro? visto que tem que ser diferenciavel??
    else:
        hubber_loss_fn = nn.SmoothL1Loss()
        loss = hubber_loss_fn(rewards.unsqueeze(1) + discount_factor*(1-terminated).unsqueeze(1)*max_q_target, model_q_a_values)

    model.optimizer.zero_grad() # zero the gradient buffers
    loss.backward()
    model.optimizer.step() # does the update
    return loss


def main_breakout(testing=False, checkpoint=None, optional_name="normal run"):
    device = torch.device("cuda")
    if testing:
        env = gym.make('ALE/Breakout-v5', render_mode='human')
    else:
        env = gym.make('ALE/Breakout-v5')
    env = BreakoutEnv(env, stack_size=4)

    last_observation, info = env.reset()
    
    lr = 2.5e-4
    m = ConvModel(last_observation.shape, env.action_space.n, lr=lr).to(device)

    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint))
        m.eval() # ?????  

    target_m = ConvModel(last_observation.shape, env.action_space.n).to(device)
    target_m.load_state_dict(m.state_dict())

    rb = ReplayBuffer(buffer_size=400000)


    # linear decay greedy
    min_rb_size = 10000 # minimum buffer to start training
    sample_size = 32
    env_steps_before_traing = 4
    target_model_update = 1000 # atualiza a target a cada N TRAINING steps, isto é N epochs, em raw steps: N*env_steps_before_traing

    # linear decay greedy
    init_epsilon = 1
    min_epsilon = 0.05
    decay_episodes = 1000000 # steps necessary to reach min_epsilon

    last_episodes_length = 50
    last_episodes_rewards = deque(maxlen=last_episodes_length) # episodios antigos sao removidos para dara lugar aos novos, assim mantem se track sempre dos ultimos
    current_episode_reward = 0.0
    
    discount_factor = 0.99
    
    steps_to_video_and_model_save = 300000

    log_steps = 1000

    import wandb
    
    if not testing:
        run = wandb.init(
            entity="franciscolaranjo9-personal",
            project="dqn-breakout",
            name=optional_name,
            config = {"min_rb_size" : min_rb_size, 
                    "sample_size" : sample_size,
                    "env_steps_before_traing" : env_steps_before_traing,
                    "target_model_update" : target_model_update,
                    "init_epsilon" : init_epsilon,
                    "min_epsilon" : min_epsilon,
                    "decay_episodes" : decay_episodes,
                    "last_episodes_length": last_episodes_length,
                    "discount_factor": discount_factor,
                    "lr" : lr}
        )


    tq = tqdm()
    try:
        steps_since_train = 0
        steps_since_target = 0
        env_steps = min_rb_size # para comecar do primeiro treino loss
        while True:
            if testing:
                env.render()
                # time.sleep(0.01)
            tq.update(1)
            if testing:
                epsilon = 0
            else:
                epsilon = min(max(env_steps*(min_epsilon - init_epsilon)/decay_episodes + init_epsilon, min_epsilon),init_epsilon) # clip para estar entre limites, linear greedy epsilon decay

            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                last_observation_tensor = torch.from_numpy(last_observation).unsqueeze(0).to(device, non_blocking=True).float()
                with torch.no_grad():
                    best_q_action = m(last_observation_tensor).detach().max(-1).indices.item()
                action = best_q_action

            observation, reward, terminated, truncated, info = env.step(action)
            current_episode_reward += reward
            
            rb.insert(Sars(last_observation, action, reward, observation, terminated, truncated))

            last_observation = observation

            if terminated or truncated:
                if not testing: 
                    run.log({"terminated_episode_reward": current_episode_reward}, step=env_steps) # to check if episodes are ending too much or not, along with their reward
                if testing:
                    print(f"ended episode reward: {current_episode_reward}")
                last_episodes_rewards.append(current_episode_reward)
                last_observation, info = env.reset()
                current_episode_reward = 0

            if env_steps % steps_to_video_and_model_save == 0:
                rew, frames = run_test_episode(target_m,BreakoutEnv(gym.make('ALE/Breakout-v5'), stack_size=4), max_steps=5000)
                frames = frames.transpose(0, 3, 1, 2)
                # # frames = (frames * 255).astype(np.uint8)
                # frames = frames[:, None, :, :]
                # frames = np.repeat(frames, 3, axis=1)  # (T,3,H,W)
                run.log({"test_reward": rew, "test_video": wandb.Video(frames, caption=str(rew), fps=25, format="mp4")})
                torch.save(target_m.state_dict(),f"models/breakout/{env_steps}.pth")

            steps_since_train += 1
            env_steps += 1
            if (not testing) and rb.size >= min_rb_size and steps_since_train >= env_steps_before_traing:
                steps_since_target += 1
                # treinar quando respeita buffer minimo e a cada env_steps_before_traing passos
                loss = train_step(m, rb, target_m, discount_factor, use_huber_loss=True)
                if env_steps % log_steps == 0:
                    run.log({"loss": loss.detach().item(), "epsilon": epsilon, "last_episodes_reward": np.mean(last_episodes_rewards)}, step=env_steps) # wandb
                steps_since_train = 0
                episode_rewards = []
                if steps_since_target >= target_model_update:
                    print("updating target_model")
                    target_m.load_state_dict(m.state_dict())
                    steps_since_target = 0
            # print(steps_since_train)
    except KeyboardInterrupt:
        pass
    env.close()
    run.finish()


main_breakout(testing=False, optional_name="HypyerTuning")
