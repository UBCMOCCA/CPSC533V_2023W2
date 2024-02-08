import random
import collections
import torch

from eval_policy import device

Transition = collections.namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer():
    def __init__(self, max_size=10000):
        self.data = collections.deque([], maxlen=max_size)

    def push(self, state, action, next_state, reward, done):
        transition = Transition(state, action, next_state, reward, done)
        self.data.append(transition)

    def sample(self, batch_size):
        transitions = random.sample(self.data, batch_size)

        batch = Transition(*zip(*transitions)) # from batch of Transitions to Transition of batches
        state = torch.tensor(batch.state, device=device, dtype=torch.float32)
        next_state = torch.tensor(batch.next_state, device=device, dtype=torch.float32)
        action = torch.tensor(batch.action, device=device, dtype=torch.float32)[:, None]
        reward = torch.tensor(batch.reward, device=device, dtype=torch.float32)
        done = torch.tensor(batch.done, device=device, dtype=torch.float32)
        return state, action, next_state, reward, done

    def __len__(self):
        return len(self.data)