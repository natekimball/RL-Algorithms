import random
from collections import deque, namedtuple

import jax
import jax.numpy as jnp

class KeyGenerator:
    def __init__(self, seed: int):
        self._key = jax.random.PRNGKey(seed)

    def __call__(self, num: int = 1) -> jax.Array:
        self._key, *subkeys = jax.random.split(self._key, num+1)
        return jnp.array(subkeys)
    

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, item: object):
        self.memory.append(item)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def flush(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)