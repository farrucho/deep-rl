from typing import Any, SupportsFloat
import numpy as np
from dataclasses import dataclass

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
    # pag 285/286 "grokking DRL" explica bem
    # uma metrica boa é ver o average score, (tmb range min e max) dos ultimos N replays, para perceber se modelo está a conseguir ter boas rewards ou nao 
    def __init__(self, buffer_size=100000, batch_size = 64):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        # ao inves de array melhor alternativa pode ser usar deque (double ended queue) from collections
        self.buffer = []

        self._index, self.size = 0, 0 # next index to modify and filled buffer size, nota o _ para descrever iterador

    def insert(self, sars: Sars):
        # sars (aka experience tuple) :  (state, action, next reward, next state)
        try:
            self.buffer[self._index] = sars
        except:
            self.buffer.append(sars)

        self._index += 1
        self._index = self._index % self.buffer_size # if _index == buffer_size then turn to 0, old experiences get overlapped by new ones, this is the definition of the replay buffer
        self.size += 1
        self.size = min(self.size, self._index)
    
    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        rng = np.random.default_rng()
        return rng.choice(self.buffer, batch_size)
    
    def __len__(self):
        return self.size
        