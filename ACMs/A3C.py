import gymnasium as gym
import os
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
from collections import deque


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]

                state['step'] = torch.zeros(1)
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()




class A3C():
    def __init__(self, params, gym_env_name, num_workers, wand_name):
        self.params = params
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda")

        self.gym_env_name = gym_env_name
        # self.env = gym.make(gym_env_name)
        self.env = AtariEnv(gym.make(gym_env_name), stack_size=4)

        last_observation, info = self.env.reset()
        self.observation_shape = last_observation.shape
        self.num_actions = self.env.action_space.n # type: ignore

        # self.global_policy_model = PolicySimpleReinforceModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device)
        # self.global_state_model = ValueSimpleStateModel(self.observation_shape, lr=self.params["lr"]).to(self.device)
        self.global_policy_model = PolicyReinforceModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device)
        self.global_state_model = ValueStateModel(self.observation_shape, lr=self.params["lr"]).to(self.device)


        self.global_policy_model.share_memory()
        self.global_state_model.share_memory()

        self.policy_optimizer = SharedAdam(
            self.global_policy_model.parameters(),
            lr=self.params["lr"]
        )

        self.state_optimizer = SharedAdam(
            self.global_state_model.parameters(),
            lr=self.params["lr"]
        )

        self.num_workers = num_workers

        self.get_out_signal = False

        self.wandb_name = wand_name
        # self.run = wandb.init(
        #     entity="franciscolaranjo9-personal",
        #     project="A3C",
        #     name=wand_name,
        #     config = self.params
        # )

        self._steps = mp.Value('i', 0)
        self._episodes = mp.Value('i', 0)

        self.log_episode_k = self.params["log_episode_k"]
        self.save_episode_k = self.params["save_episode_k"]
        self.deque_test_rewards = deque(maxlen=self.params["deque_test_rewards"])

        self.log_queue = mp.Queue()
    
    def log_process(self):
        print("log process ready")
        run = wandb.init(
            entity="franciscolaranjo9-personal",
            project="A3C",
            name=self.wandb_name,
            config=self.params
        )

        while True:
            log_dict = self.log_queue.get()
            run.log(log_dict)

    
    def generate_episode(self, local_env, last_observation, policy_model, state_values_model, max_steps=1000000):
        rewards = []
        logprobs = []
        state_values = []
        entropies = []
        terminated, truncated = False, False
        _idx = 0
        frames = []
        exit_loop = False

        bootstrap_value = 0
        while not truncated and not exit_loop:
            last_observation_tensor = torch.from_numpy(last_observation).to(self.device).unsqueeze(0).float()


            policy_actions = policy_model(last_observation_tensor)
            softmax_probs = nn.functional.softmax(policy_actions, dim=-1)
            dist = torch.distributions.Categorical(softmax_probs)
            action = dist.sample() # retorna index
            
            state_value = state_values_model(last_observation_tensor)
            logprob = dist.log_prob(action)
            entropy = dist.entropy() # this uses ln(6) - torch.sum(softmax_probs * torch.log2(softmax_probs), dim=-1) 



            observation, reward, terminated, truncated, info = local_env.step(action.item())

            if terminated:
               bootstrap_value = 0
               last_observation, _ = local_env.reset()
               exit_loop = True 
            if _idx == max_steps:
                bootstrap_value = state_values_model(torch.from_numpy(observation).unsqueeze(0).to(self.device).float()).detach().item()
                exit_loop = True

            last_observation = observation
            rewards.append(reward)
            logprobs.append(logprob)
            state_values.append(state_value)
            entropies.append(entropy)
            # frames.append(info["rgb_frame"])
            frames.append(0)
            _idx += 1
            
            with self._steps.get_lock():
                self._steps.value += 1
        
        return local_env, last_observation, torch.tensor(rewards).unsqueeze(1), torch.stack(logprobs), torch.stack(state_values).squeeze(1), torch.stack(entropies), bootstrap_value, np.stack(frames, 0)


    def work(self, rank):
        print(f"Worker {rank} at your service")
        
        # local_env = gym.make(self.gym_env_name)
        local_env = AtariEnv(gym.make(self.gym_env_name), stack_size=4)
        last_observation, info = local_env.reset()
        _episodes = 0
        
        local_policy_model = PolicyReinforceModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device) # TODO METER ISTO EM CIMA CAUSA BUG E ENTROPIA COLAPSA LOGO???
        local_value_model = ValueStateModel(self.observation_shape, lr=self.params["lr"]).to(self.device)
        
        while not self.get_out_signal:
            local_policy_model.load_state_dict(self.global_policy_model.state_dict())
            local_value_model.load_state_dict(self.global_state_model.state_dict())


            # ---generate episode ---
            local_env, last_observation, rewards, logprobs, state_values, entropies, bootstrap_value, frames_of_episode = self.generate_episode(local_env, last_observation, local_policy_model, local_value_model, max_steps=self.params["n_step_bootstrapping"])

            discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
            G = bootstrap_value
            for j in reversed(range(0,len(rewards))):
                G = rewards[j] + G*self.params["discount_factor"]
                discounted_rewards[j] = G



            # --- calculate loss---
            local_state_model_loss = torch.mean(torch.pow(discounted_rewards - state_values,2)) # do not normalize state values

            advantage_term_policy = discounted_rewards - state_values.detach()
            advantage_term_policy = (advantage_term_policy - advantage_term_policy.mean()) / (advantage_term_policy.std() + 1e-8) # normalize, in this case the mean will be 0

            local_policy_model_loss = -torch.mean((advantage_term_policy)*logprobs) - self.params["beta"]*torch.mean(entropies) # detach aqui é essencial caso contrario a backprop vai para o value model, coisa que não queremos, pois so estamos a dar update ao step model


            # --- optimize ---
            self.policy_optimizer.zero_grad()
            local_policy_model_loss.backward()

            # copy gradients from the local model to shared model
            for param, shared_param in zip(local_policy_model.parameters(), self.global_policy_model.parameters()):
                # if shared_param.grad is None: # only push when global_policy has learned once
                shared_param._grad = param.grad

            nn.utils.clip_grad_norm_(local_policy_model.parameters(), self.params["clip_grad"])

            self.policy_optimizer.step()
            local_policy_model.load_state_dict(self.global_policy_model.state_dict())


            self.state_optimizer.zero_grad()
            local_state_model_loss.backward()
            nn.utils.clip_grad_norm_(local_value_model.parameters(), self.params["clip_grad"])

            # copy gradients from the local model to shared model
            for param, shared_param in zip(local_value_model.parameters(), self.global_state_model.parameters()):
                # if shared_param.grad is None: # only push when global_policy has learned once
                shared_param._grad = param.grad

            self.state_optimizer.step()
            local_value_model.load_state_dict(self.global_state_model.state_dict())

            _episodes += 1
            with self._episodes.get_lock():
                self._episodes.value += 1


            if rank == 0 and _episodes % self.log_episode_k == 0:
                # test_env = gym.make(self.gym_env_name)
                test_env = AtariEnv(gym.make(self.gym_env_name), stack_size=4)
                test_env_observation, info = test_env.reset()
                test_env, test_env_observation, test_rewards, test_logprobs, test_state_values, test_entropies, test_bootstrap_value, frames_of_episode = self.generate_episode(test_env, test_env_observation, self.global_policy_model, self.global_state_model, max_steps=100000000)
                total_reward = test_rewards.sum().item()
                self.deque_test_rewards.append(total_reward)
                self.log_queue.put(
                    {f"full_episode_reward" : total_reward, \
                        f"mean_entropy" : torch.mean(test_entropies).item(), \
                                f"total_episodes" : self._episodes.value, \
                                    f"deque_test_rewards": np.mean(self.deque_test_rewards), \
                                        f"total_step": self._steps.value}
                )

            if rank == 0 and _episodes % self.save_episode_k == 0:
                try:
                    os.makedirs(f"models/A3C/{self.gym_env_name}/{self.wandb_name}/policy_model/")
                    os.makedirs(f"models/A3C/{self.gym_env_name}/{self.wandb_name}/value_model/")
                except:
                    torch.save(self.global_policy_model.state_dict(),f"models/A3C/{self.gym_env_name}/{self.wandb_name}/policy_model/{self._episodes.value}.pth")
                    torch.save(self.global_state_model.state_dict(),f"models/A3C/{self.gym_env_name}/{self.wandb_name}/value_model/{self._episodes.value}.pth")

            self.log_queue.put(
                {f"rank_{rank}/generated_episode_reward" : rewards.sum().item(), \
                    f"rank_{rank}/policy_model_loss" : local_policy_model_loss.item(), \
                        f"rank_{rank}/state_model_loss" : local_state_model_loss.item(), \
                            f"rank_{rank}/mean_entropy" : torch.mean(entropies).item(), \
                                f"rank_{rank}/advantage_term_policy" : torch.mean(advantage_term_policy).item(), \
                                    f"rank_{rank}/episode" : _episodes}
            )
        
    def train(self):
        log_p = mp.Process(target=self.log_process)
        log_p.start()
        processes = []
        for j in range(0, self.num_workers):
            p = mp.Process(target=self.work, args=(j,))
            # import pickle

            # for attr_name, attr_value in self.__dict__.items():
            #     try:
            #         pickle.dumps(attr_value)
            #     except Exception as e:
            #         print(f"Attribute '{attr_name}' is NOT picklable! Error: {e}")
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        log_p.join()
         



params = {
    "lr": 1e-4,
    "discount_factor": 0.99,
    "beta": 0.01,
    "n_step_bootstrapping": 15,
    "clip_grad": 40,

    # non related to algorithm
    "log_episode_k" : 10,
    "deque_test_rewards" : 20,
    "save_episode_k": 550 # not global episode, rank0 worker episodes
}

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # a3c = A3C(params, gym_env_name="CartPole-v1", num_workers=16 , wand_name="Testing Cartpole 16 workers")
    a3c = A3C(params, gym_env_name="PongNoFrameskip-v4", num_workers=8, wand_name="Tuning Pong 8 workers")
    
    a3c.train()
