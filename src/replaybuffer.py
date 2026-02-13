from typing import Any, SupportsFloat
import numpy as np
from dataclasses import dataclass
from collections import deque
import torch

# https://realpython.com/ref/stdlib/dataclasses/
@dataclass
class Sars:
    state: Any
    action: Any
    reward: SupportsFloat
    next_state: Any
    terminated: Any
    truncated: Any



class ReplayBuffer:
    # isto está super otimizado :)
    # pag 285/286 "grokking DRL" explica bem
    # uma metrica boa é ver o average score, (tmb range min e max) dos ultimos N replays, para perceber se modelo está a conseguir ter boas rewards ou nao 
    def __init__(self, buffer_size=100000, batch_size=64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # ao inves de array melhor alternativa pode ser usar deque (double ended queue) from collections
        # self.buffer = []

        self._index, self.size = 0, 0 # next index to modify and filled buffer size, nota o _ para descrever iterador

        self.buffer_opened = False
        self.current_states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.terminated = None
        self.truncated = None
        


    def insert(self, sars: Sars):
        # sars (aka experience tuple) :  (state, action, next reward, next state)
        if not self.buffer_opened:
            # self.buffer[self._index] = sars
            self.current_states = np.empty((self.buffer_size, *sars.state.shape), dtype=np.uint8)
            self.actions = np.empty((self.buffer_size), dtype=np.int8)
            self.rewards = np.empty((self.buffer_size), dtype=np.float32)
            self.next_states = np.empty((self.buffer_size, *sars.next_state.shape), dtype=np.uint8)
            self.terminated = np.empty((self.buffer_size), dtype=np.int8)
            self.truncated = np.empty((self.buffer_size), dtype=np.int8)
            self.buffer_opened = True

            total_mem = (self.current_states.nbytes + self.next_states.nbytes +  self.rewards.nbytes + self.actions.nbytes) / (1024**3) # os outros ocupam mb e é pouco
            print(f"Replay Buffer is going to occupy {total_mem:.2f} GB")
        


        # self.buffer.append(sars)
        self.current_states[self._index] = (sars.state * 255).astype(np.uint8)
        self.actions[self._index] = sars.action
        self.rewards[self._index] = sars.reward
        self.next_states[self._index] = (sars.next_state * 255).astype(np.uint8)
        self.terminated[self._index] = 1 if sars.terminated else 0
        self.truncated[self._index] = 1 if sars.truncated else 0

        self._index += 1
        self._index = self._index % self.buffer_size # if _index == buffer_size then turn to 0, old experiences get overlapped by new ones, this is the definition of the replay buffer
        self.size += 1
        self.size = min(self.size, self.buffer_size)
    
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        indices = np.random.randint(0, self.size, size=batch_size)

        device = torch.device("cuda")
        
        current_states = torch.from_numpy(self.current_states[indices]).to(device, non_blocking=True).float() / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(device, non_blocking=True).long().unsqueeze(1)
        rewards = torch.from_numpy(self.rewards[indices]).to(device, non_blocking=True)
        next_states = torch.from_numpy(self.next_states[indices]).to(device, non_blocking=True).float() / 255.0
        terminated = torch.from_numpy(self.terminated[indices]).to(device, non_blocking=True)
        truncated = torch.from_numpy(self.truncated[indices]).to(device, non_blocking=True)

        return current_states, actions, rewards, next_states, terminated, truncated
    
    def __len__(self):
        return self.size
        