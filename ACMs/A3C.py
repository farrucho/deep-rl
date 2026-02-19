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


class A3C():
    def __init__(self, params, gym_env_name, num_workers, wand_name, log_episode_k):
        self.params = params
        self.device = torch.device("cpu")

        self.gym_env_name = gym_env_name
        self.env = gym.make(gym_env_name)
        last_observation, info = self.env.reset()
        self.observation_shape = last_observation.shape
        self.num_actions = self.env.action_space.n # type: ignore

        self.global_policy_model = PolicySimpleReinforceModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device)
        self.global_state_model = ValueSimpleStateModel(self.observation_shape, lr=self.params["lr"]).to(self.device)

        self.global_policy_model.share_memory()
        self.global_state_model.share_memory()

        self.num_workers = num_workers

        self.get_out_signal = False

        self.run = wandb.init(
            entity="franciscolaranjo9-personal",
            project="A3C",
            name=wand_name,
            config = self.params
        )
        self._steps = 0
        self._episodes = 0
        self.log_episode_k = log_episode_k
    
    
    def generate_episode(self, policy_model, state_values_model, max_steps=1000000):
        # TODO ESTA ERRADO POIS SO GERA EPISODIO ATE N_sTEPBOOSTRAPING DEPOIS PAROU
        local_env = gym.make(self.gym_env_name)
        last_observation, info = local_env.reset()
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
            self._steps += 1



        return torch.tensor(rewards).unsqueeze(1), torch.stack(logprobs), torch.stack(state_values).squeeze(1), torch.stack(entropies), bootstrap_value, np.stack(frames, 0)


    def work(self, rank):
        print(f"Worker {rank} at your service")
        
        _episodes = 0
        while not self.get_out_signal:
            local_policy_model = PolicySimpleReinforceModel(self.observation_shape, self.num_actions, lr=self.params["lr"]).to(self.device) # TODO METER ISTO EM CIMA CAUSA BUG E ENTROPIA COLAPSA LOGO???
            local_value_model = ValueSimpleStateModel(self.observation_shape, lr=self.params["lr"]).to(self.device)

            local_policy_model.load_state_dict(self.global_policy_model.state_dict())
            local_value_model.load_state_dict(self.global_state_model.state_dict())


            # ---generate episode ---
            rewards, logprobs, state_values, entropies, bootstrap_value, frames_of_episode = self.generate_episode(local_policy_model, local_value_model, max_steps=self.params["n_step_bootstrapping"])

            discounted_rewards = torch.zeros_like(rewards, dtype=torch.float32, device=self.device)
            G = bootstrap_value
            for j in reversed(range(0,len(rewards))):
                G = rewards[j] + G*self.params["discount_factor"]
                discounted_rewards[j] = G



            # --- calculate loss---
            local_state_model_loss = torch.mean(torch.pow(discounted_rewards - state_values,2)) # do not normalize state values

            advantage_term_policy = discounted_rewards - state_values.detach()
            # advantage_term_policy = (advantage_term_policy - advantage_term_policy.mean()) / (advantage_term_policy.std() + 1e-8) # normalize, in this case the mean will be 0
            local_policy_model_loss = -torch.mean((advantage_term_policy)*logprobs + params["beta"]*entropies) # detach aqui é essencial caso contrario a backprop vai para o value model, coisa que não queremos, pois so estamos a dar update ao step model


            # --- optimize --- do i need lock here? # TODO
            self.global_policy_model.zero_grad()
            local_policy_model_loss.backward()
            # TODO clip gradientes? nn.utils.clip_grad_norm_(local_policy_model.parameters(), policy_max_grad_norm)

            # copy gradients from the local model to shared model
            for param, shared_param in zip(local_policy_model.parameters(), self.global_policy_model.parameters()):
                if shared_param.grad is None: # only push when global_policy has learned once
                    shared_param._grad = param.grad

            self.global_policy_model.optimizer.step()
            local_policy_model.load_state_dict(self.global_policy_model.state_dict())


            self.global_state_model.zero_grad()
            local_state_model_loss.backward()
            # TODO clip gradientes? nn.utils.clip_grad_norm_(local_policy_model.parameters(), policy_max_grad_norm)

            # copy gradients from the local model to shared model
            for param, shared_param in zip(local_value_model.parameters(), self.global_state_model.parameters()):
                if shared_param.grad is None: # only push when global_policy has learned once
                        shared_param._grad = param.grad

            self.global_state_model.optimizer.step()
            local_value_model.load_state_dict(self.global_state_model.state_dict())

            # rank 0 is the responsible for logging test episodes to see episode generated by global policy  
            if rank == 0 and self._episodes % self.log_episode_k == 0:
                rewards, logprobs, state_values, entropies, bootstrap_value, frames_of_episode = self.generate_episode(self.global_policy_model, self.global_state_model, max_steps=self.params["n_step_bootstrapping"])
                self.run.log({f"episode_reward" : rewards.sum().item(), \
                                f"mean_entropy" : torch.mean(entropies).item(), \
                                        f"total_episodes" : self._episodes})


            self.run.log({f"rank_{rank}/episode_reward" : rewards.sum().item(), \
                          f"rank_{rank}/policy_model_loss" : local_policy_model_loss, \
                            f"rank_{rank}/state_model_loss" : local_state_model_loss, \
                                f"rank_{rank}/mean_entropy" : torch.mean(entropies).item(), \
                                    f"rank_{rank}/advantage_term_policy" : torch.mean(advantage_term_policy).item(), \
                                        f"rank_{rank}/episode" : _episodes})
            _episodes += 1
            self._episodes += 1
        
    def train(self):
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
         


params = {
    "lr": 1e-4,
    "n_step_bootstrapping": 1000,
    "discount_factor": 0.99,
    "beta": 1e-3,
    "n_step_bootstrapping": 300,
}
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    a3c = A3C(params, gym_env_name="CartPole-v1", num_workers=16 , wand_name="Cartpole 16 workers", log_episode_k=50)

    a3c.train()

    # TODO
    # atualmente cada episodio apenas corre ate terminar OU n_steps, se nstepboostrapping for pequeno o episodio nao corre ate ao final fica limitiado,ver como se faz a implemetnacao correta!
    # SHARED ADAM OPTIMIZER,
    # - clip gradient, 
    # - ver bug de criaar model a cada episodio .