import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, buffer_size, observation_shape, action_dim, device=None):
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.observations = np.zeros((buffer_size, *observation_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        self.pos = 0
        self.full = False
        
    def add(self, obs, action, reward, value, log_prob, done):
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        self.dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            
    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        last_gae_lam = 0
        for step in reversed(range(self.pos)):
            if step == self.pos - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
            
        self.returns = self.advantages + self.values
        
    def get_samples(self, batch_size=None):
        indices = np.arange(self.pos if not self.full else self.buffer_size)
        np.random.shuffle(indices)
        
        if batch_size is None:
            batch_size = len(indices)
            
        for start_idx in range(0, len(indices), batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                torch.as_tensor(self.observations[batch_indices], device=self.device),
                torch.as_tensor(self.actions[batch_indices], device=self.device),
                torch.as_tensor(self.returns[batch_indices], device=self.device),
                torch.as_tensor(self.advantages[batch_indices], device=self.device),
                torch.as_tensor(self.log_probs[batch_indices], device=self.device)
            )
            
    def clear(self):
        self.pos = 0
        self.full = False