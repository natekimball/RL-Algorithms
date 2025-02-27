import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import namedtuple
from gymnasium import wrappers

class NeuralNet(nn.Module):
    def __init__(self, obs_dim, action_space, hidden_size=128):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.logits = nn.Linear(hidden_size, action_space)
        self.value = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.xavier_normal_(self.value.weight)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.logits(x)
        value = self.value(x)
        return logits, value

gamma = 0.99
lmda = 0.95
epsilon = 0.2
seed = 0
lr = 0.001
ppo_epochs = 4

env = gym.make('CartPole-v1')
print('observation shape:', env.observation_space.shape)
print('action shape:', env.action_space.shape)

neural_net = NeuralNet(obs_dim=env.observation_space.shape[0], action_space=env.action_space.n)
optimizer = optim.Adam(neural_net.parameters(), lr=lr)

Transition = namedtuple('Transition',
                        ['obs', 'act', 'rew', 'next_obs', 'terminated', 'logits'])

def policy(params, obs):
    '''
    Given the current state, return the action and the logits.
    '''
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    logits, _ = params(obs)
    action_dist = torch.distributions.Categorical(logits=logits)
    action = action_dist.sample()
    return action.item(), logits.detach()

def unroll_policy(params, T=100):
    '''
    Unroll the policy for T steps and return the trajectory as a list of transitions.
    '''
    trajectory = []
    obs, _ = env.reset()
    for _ in range(T):
        action, logits = policy(params, obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        trajectory.append(Transition(obs, action, reward, next_obs, terminated, logits))
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return trajectory

def advantage_estimation(trajectory, params):
    '''
    Given a trajectory, estimate the advantages using truncated GAE.
    '''
    obs = torch.tensor(np.array([t.obs for t in trajectory]), dtype=torch.float32)
    next_obs = torch.tensor(np.array([t.next_obs for t in trajectory]), dtype=torch.float32)
    rewards = torch.tensor(np.array([t.rew for t in trajectory]), dtype=torch.float32)
    terminated = torch.tensor(np.array([t.terminated for t in trajectory]), dtype=torch.float32)
    
    _, values = params(obs)
    _, next_values = params(next_obs)
    values, next_values = values.squeeze(), next_values.squeeze()
    deltas = rewards + gamma * next_values * (1 - terminated) - values
    advantages = torch.zeros_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(len(trajectory) - 1)):
        advantages[t] = deltas[t] + gamma * lmda * (1 - terminated[t]) * advantages[t + 1]
    return advantages

def policy_loss(params, obs, actions, old_logits, advantages):
    logits, _ = params(obs)
    log_probs = nn.functional.log_softmax(logits, dim=1)
    old_log_probs = nn.functional.log_softmax(old_logits.reshape(-1,2), dim=1)
    log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    old_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    surrogate = torch.min(ratio * advantages, clipped_ratio * advantages)
    return -torch.mean(surrogate)

def value_loss(params, obs, next_obs, rewards, terminated):
    _, values = params(obs)
    _, next_values = params(next_obs)
    values, next_values = values.squeeze(), next_values.squeeze()
    tde = rewards + gamma * next_values * (1 - terminated) - values
    return torch.mean(tde ** 2)

def ppo_loss(params, obs, actions, old_logits, advantages, next_obs, rewards, terminated, loss_ratio=0.5):
    p_loss = policy_loss(params, obs, actions, old_logits, advantages)
    v_loss = value_loss(params, obs, next_obs, rewards, terminated)
    return p_loss + loss_ratio * v_loss

def update(params, optimizer, traj_segment, advantage_segment):
    obs = torch.tensor([t.obs for t in traj_segment], dtype=torch.float32)
    actions = torch.tensor([t.act for t in traj_segment], dtype=torch.int64)
    old_logits = torch.stack([t.logits for t in traj_segment])
    advantages = advantage_segment.detach()
    next_obs = torch.tensor([t.next_obs for t in traj_segment], dtype=torch.float32)
    rewards = torch.tensor([t.rew for t in traj_segment], dtype=torch.float32)
    terminated = torch.tensor([t.terminated for t in traj_segment], dtype=torch.int32)
    
    optimizer.zero_grad()
    loss = ppo_loss(params, obs, actions, old_logits, advantages, next_obs, rewards, terminated, loss_ratio=0.5)
    loss.backward()
    optimizer.step()

def evaluate(params, num_eps=10):
    env_eval = wrappers.RecordEpisodeStatistics(env)
    cum_rews = []
    for _ in range(num_eps):
        obs, _ = env_eval.reset()
        terminated, truncated = False, False
        while not terminated and not truncated:
            action, _ = policy(params, obs)
            obs, _, terminated, truncated, info = env_eval.step(action)
        cum_rews.append(info['episode']['r'])
    return np.mean(cum_rews)

steps = 150
unroll_size = 1280
mini_size = 128
eval_freq = 10
best_params = neural_net.state_dict()
best_mean_ret = 0
eval_steps = []
mean_returns = []

for step in range(1, steps + 1):
    traj = unroll_policy(neural_net, T=unroll_size)
    advantage = advantage_estimation(traj, neural_net)

    for _ in range(ppo_epochs):
        # Shuffle data for better learning
        indices = np.arange(unroll_size)
        np.random.shuffle(indices)
        
        # Process mini-batches
        for start in range(0, unroll_size, mini_size):
            end = start + mini_size
            if end <= unroll_size:
                idx = indices[start:end]
                mini_traj = [traj[i] for i in idx]
                mini_adv = advantage[idx]
                update(neural_net, optimizer, mini_traj, mini_adv)

    # for _ in range(ppo_epochs):
    #     for i in range(0, unroll_size, mini_size):
    #         mini_traj = traj[i:i + mini_size]
    #         mini_adv = advantage[i:i + mini_size]
    #         update(neural_net, optimizer, mini_traj, mini_adv)

    if step % eval_freq == 0:
        mean_ret = evaluate(neural_net)
        eval_steps.append(step)
        mean_returns.append(mean_ret)
        if mean_ret > best_mean_ret:
            best_mean_ret = mean_ret
            best_params = neural_net.state_dict()
        print(f'step: {step}, mean return: {mean_ret}')
        torch.save({'model_state_dict': best_params}, f'checkpoints/ckpt_{step}.pth')

torch.save({'model_state_dict': best_params}, 'checkpoints/ckpt_last.pth')
plt.figure()
plt.plot(eval_steps, mean_returns, label='mean returns')
plt.legend()
plt.savefig('mean_returns.png')
env.close()

# Generate video
test_env = gym.make('CartPole-v1', render_mode='rgb_array', max_episode_steps=500)
test_env = wrappers.RecordVideo(env=test_env, video_folder='./video', name_prefix='PPO', disable_logger=True)

obs, _ = test_env.reset(seed=42)
terminated, truncated = False, False
while not terminated and not truncated:
    action, _ = policy(neural_net, obs)
    obs, _, terminated, truncated, _ = test_env.step(action)
test_env.close()


# import os
# import pickle
# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# from collections import namedtuple
# from gymnasium import wrappers
# from utils.common import KeyGenerator


# class NeuralNet(nn.Module):
#     def __init__(self, input_dim, hidden_size=128, action_space=2):
#         super(NeuralNet, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.logits = nn.Linear(hidden_size, action_space)
#         self.value = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         logits = self.logits(x)
#         value = self.value(x)
#         return logits, value

# gamma = 0.99
# lmda = 0.95
# epsilon = 0.2
# seed = 0
# lr = 0.0003
# n_epochs = 4
# value_coeff = 0.5
# entropy_coeff = 0.01

# torch.manual_seed(seed)

# env = gym.make('CartPole-v1')
# input_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n

# neural_net = NeuralNet(input_dim, action_space=action_dim)
# optimizer = optim.Adam(neural_net.parameters(), lr=lr)
# Transition = namedtuple('Transition', ['obs', 'act', 'rew', 'next_obs', 'terminated', 'logits'])
# obs, _ = env.reset(seed=seed)

# def policy(params, obs):
#     obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
#     logits, _ = params(obs)
#     action = torch.distributions.Categorical(logits=logits).sample()
#     return action.item(), logits.detach().numpy()

# def unroll_policy(params, T=100):
#     trajectory = []
#     obs, _ = env.reset()
#     for _ in range(T):
#         action, logits = policy(params, obs)
#         next_obs, reward, terminated, truncated, _ = env.step(action)
#         trajectory.append(Transition(obs, action, reward, next_obs, terminated, logits))
#         obs = next_obs if not (terminated or truncated) else env.reset()[0]
#     return trajectory

# def compute_advantages(trajectory, params):
#     obs = torch.tensor([t.obs for t in trajectory], dtype=torch.float32)
#     next_obs = torch.tensor([t.next_obs for t in trajectory], dtype=torch.float32)
#     rewards = torch.tensor([t.rew for t in trajectory], dtype=torch.float32)
#     terminated = torch.tensor([t.terminated for t in trajectory], dtype=torch.float32)
    
#     with torch.no_grad():
#         _, values = params(obs)
#         _, next_values = params(next_obs)
#         values, next_values = values.squeeze(), next_values.squeeze()
        
#         deltas = rewards + gamma * next_values * (1 - terminated) - values
#         advantages = torch.zeros_like(rewards)
#         adv = 0
#         for t in reversed(range(len(trajectory))):
#             adv = deltas[t] + gamma * lmda * (1 - terminated[t]) * adv
#             advantages[t] = adv
        
#         # Normalize advantages - critical for stable training
#         advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
#     returns = advantages + values  # For value function training
#     return advantages, returns

# def ppo_loss(params, batch, advantages, returns):
#     obs = torch.tensor(batch['obs'], dtype=torch.float32)
#     actions = torch.tensor(batch['actions'], dtype=torch.long)
#     old_logits = torch.tensor(batch['logits'], dtype=torch.float32)
    
#     logits, values = params(obs)
#     values = values.squeeze()
    
#     # Policy loss with proper log prob calculation
#     log_probs = nn.functional.log_softmax(logits, dim=1)
#     old_log_probs = nn.functional.log_softmax(old_logits.reshape(-1,2), dim=1)

#     log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
#     old_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
    
#     ratio = torch.exp(log_probs - old_log_probs)
#     surrogate1 = ratio * advantages
#     surrogate2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
#     policy_loss = -torch.min(surrogate1, surrogate2).mean()
    
#     # Value loss
#     value_loss = value_coeff * ((values - returns) ** 2).mean()
    
#     # Entropy bonus for exploration
#     # entropy = dist.entropy().mean()
    
#     # Total loss
#     loss = policy_loss + value_loss #- entropy_coeff * entropy
    
#     return loss, policy_loss, value_loss#, entropy

# def update(params, optimizer, traj_segment, advantages, returns):
#     batch = {
#         'obs': np.array([t.obs for t in traj_segment]),
#         'actions': np.array([t.act for t in traj_segment]),
#         'logits': np.array([t.logits for t in traj_segment]),
#     }
    
#     # loss, policy_loss, value_loss, entropy = ppo_loss(params, batch, advantages, returns)
#     loss, policy_loss, value_loss = ppo_loss(params, batch, advantages, returns)
    
#     optimizer.zero_grad()
#     loss.backward()
#     # Add gradient clipping for stability
#     torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=0.5)
#     optimizer.step()
    
#     return loss.item(), policy_loss.item(), value_loss.item()#, entropy.item()

# def evaluate(params, num_eps=10):
#     env_eval = wrappers.RecordEpisodeStatistics(env)
#     cum_rews = []
#     for _ in range(num_eps):
#         obs, _ = env_eval.reset()
#         terminated, truncated = False, False
#         while not terminated and not truncated:
#             action, _ = policy(params, obs)
#             obs, _, terminated, truncated, info = env_eval.step(action)
#         cum_rews.append(info['episode']['r'])
#     return np.mean(cum_rews)

# steps = 150
# unroll_size = 1280
# mini_size = 128
# eval_freq = 10
# best_params = neural_net.state_dict()
# best_mean_ret = 0
# eval_steps, mean_returns = [], []

# for step in range(1, steps + 1):
#     traj = unroll_policy(neural_net, T=unroll_size)
#     advantages, returns = compute_advantages(traj, neural_net)
    
#     # Multiple optimization epochs over the same data
#     for _ in range(n_epochs):
#         # Shuffle data for better learning
#         indices = np.arange(unroll_size)
#         np.random.shuffle(indices)
        
#         # Process mini-batches
#         for start in range(0, unroll_size, mini_size):
#             end = start + mini_size
#             if end <= unroll_size:
#                 idx = indices[start:end]
#                 mini_traj = [traj[i] for i in idx]
#                 mini_adv = advantages[idx]
#                 mini_ret = returns[idx]
#                 update(neural_net, optimizer, mini_traj, mini_adv, mini_ret)

#     if step % eval_freq == 0:
#         mean_ret = evaluate(neural_net)
#         eval_steps.append(step)
#         mean_returns.append(mean_ret)
#         if mean_ret > best_mean_ret:
#             best_mean_ret = mean_ret
#             best_params = neural_net.state_dict()
#         print(f'step: {step}, mean return: {mean_ret}')
#         torch.save({'model_state_dict': best_params, 'optimizer_state_dict': optimizer.state_dict()}, f'checkpoints/ckpt_{step}.pth')

# torch.save({'mel_state_dict': best_params, 'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/ckpt_last.pth')
# plt.figure()
# plt.plot(eval_steps, mean_returns, label='mean returns')
# plt.legend()
# plt.savefig('mean_returns.png')
# env.close()
