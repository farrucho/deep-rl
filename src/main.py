from tabnanny import check
from agent import DNQAgent
from replaybuffer import ReplayBuffer, Sars
from model import Model
import gymnasium as gym
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import deque
import time

# env = gym.make('CartPole-v1', render_mode='human')
# observation, info = env.reset()
# observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

# m = Model(env.observation_space.shape, env.action_space.n)

# q_values = m(torch.from_numpy(observation))
# print(q_values)


# rb = ReplayBuffer()
# # env = gym.make('CartPole-v1', render_mode='human')
# env = gym.make('CartPole-v1')
# last_observation, info = env.reset()

# try:
#     while True:
#         action = env.action_space.sample()
#         observation, reward, terminated, truncated, info = env.step(action)

#         rb.insert(Sars(last_observation, action, reward, observation, terminated, truncated))
#         last_observation = observation

#         if terminated or truncated:
#             last_observation, info = env.reset()
# except KeyboardInterrupt:
#     pass
# env.close()


def train_step(model, experiences, target_model, discount_factor=0.99):
    device = torch.device("cuda")

    current_states = torch.stack([torch.Tensor(e.state) for e in experiences]).to(device)
    actions = torch.tensor([e.action for e in experiences]).to(device).unsqueeze(1) # shape (N,1); unsqueeze:"Returns a new tensor with a dimension of size one inserted at the specified position."
    rewards = torch.tensor([torch.tensor(e.reward) for e in experiences]).to(device)
    next_states = torch.stack([torch.Tensor(e.next_state) for e in experiences]).to(device)
    terminated = torch.stack([torch.Tensor([0]) if e.terminated else torch.Tensor([1]) for e in experiences]).to(device)
    truncated = torch.stack([torch.Tensor([0]) if e.truncated else torch.Tensor([1]) for e in experiences]).to(device)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode. Also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        target_q_values_nextstate = target_model(next_states).to(device) # Q(s',a'; \theta^{-}) shape: (N, num_actions)
    
    model_q_values = model(current_states) # [Q(s,a_0; \theta_{i}), Q(s,a_1; \theta_{i}), ... , Q(s,a_num_actions; \theta_{i})] shape: (N, num_actions)
    # nos nao queremos isto, queremos ir buscar especificamente Q(s,a; \theta_{i}), temos que usar o vetor actions
    model_q_a_values = torch.gather(model_q_values, 1, actions) # isto funciona porque actions é 0 ou 1, caso contrário seria preciso one hot encoding, para cada linha do model_q_values vai buscar a coluna action (0 ou 1) respetiva, obtendo corretamente Q(s,a; \theta_{i})
    # https://docs.pytorch.org/docs/stable/generated/torch.gather.html

    max_q_target = torch.max(target_q_values_nextstate, 1, keepdim=True).values# (N,1) max values over each line (dimension to reduce is over the number of actions)
    


    # falta discount factor
    loss = torch.mean((rewards + discount_factor*terminated*max_q_target - model_q_a_values)**2)# eq. in pag. 260 Será que vai dar erro? visto que tem que ser diferenciavel??


    model.optimizer.zero_grad() # zero the gradient buffers
    loss.backward()
    model.optimizer.step() # does the update
    return loss



def main(testing=False, checkpoint=None, optional_name="linear decay greedy"):
    device = torch.device("cuda")
    if testing:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    last_observation, info = env.reset()

    m = Model(env.observation_space.shape, env.action_space.n).to(device)

    if checkpoint is not None:
        m.load_state_dict(torch.load(checkpoint))
        m.eval() # ?????

    target_m = Model(env.observation_space.shape, env.action_space.n).to(device)
    target_m.load_state_dict(m.state_dict())

    rb = ReplayBuffer()


    min_rb_size = 10000 # minimum buffer to start training
    sample_size = 300
    env_steps_before_traing = 100
    target_model_update = 30 # atualiza a target a cada N TRAINING steps, isto é N epochs, em raw steps: N*env_steps_before_traing

    # linear decay greedy
    init_epsilon = 1.0
    min_epsilon = 0.05
    decay_episodes = 100000 # steps necessaray to reach min_epsilon

    last_episodes_length = 50
    last_episodes_rewards = deque(maxlen=last_episodes_length) # episodios antigos sao removidos para dara lugar aos novos, assim mantem se track sempre dos ultimos
    current_episode_reward = 0.0
    
    discount_factor = 0.99

    import wandb
    
    if not testing:
        run = wandb.init(
            entity="franciscolaranjo9-personal",
            project="dqn-cartpole",
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
                    "lr" : 1e-4}
        )


    tq = tqdm()
    try:
        steps_since_train = 0
        steps_since_target = 0
        env_steps = 0-min_rb_size # para comecar do primeiro treino loss
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
                best_q_action = m(torch.tensor(last_observation).to(device)).detach().max(-1).indices.item()
                action = best_q_action

            observation, reward, terminated, truncated, info = env.step(action)
            current_episode_reward += reward

            rb.insert(Sars(last_observation, action, reward/100, observation, terminated, truncated)) # the reward can go up to 500 so we must kinda normalize the data to see if the nn can learn better.

            last_observation = observation

            if terminated or truncated:
                if not testing: 
                    run.log({"terminated_episode_reward": current_episode_reward}, step=env_steps) # to check if episodes are ending too much or not, along with their reward
                if testing:
                    print(f"ended episode reward: {current_episode_reward}")
                last_episodes_rewards.append(current_episode_reward)
                last_observation, info = env.reset()
                current_episode_reward = 0

            steps_since_train += 1
            env_steps += 1
            if (not testing) and rb.size >= min_rb_size and steps_since_train >= env_steps_before_traing:
                steps_since_target += 1
                # treinar quando respeita buffer minimo e a cada env_steps_before_traing passos
                loss = train_step(m, rb.sample(sample_size), target_m, discount_factor)
                run.log({"loss": loss.detach().item(), "epsilon": epsilon, "last_episodes_reward": np.mean(last_episodes_rewards)}, step=env_steps) # wandb
                steps_since_train = 0
                episode_rewards = []
                if steps_since_target >= target_model_update:
                    print("updating target_model")
                    target_m.load_state_dict(m.state_dict())
                    steps_since_target = 0
                    torch.save(target_m.state_dict(),f"models/{env_steps}.pth")
            # print(steps_since_train)
    except KeyboardInterrupt:
        pass
    env.close()
    run.finish()

# main(testing=True, checkpoint="./models/1064990.pth")
# main(testing=True, checkpoint="./models/2183900.pth")
# main(testing=False, optional_name="array with cuda double batch size 128")
# main(testing=False, optional_name="hypertuning with discountfactor")
