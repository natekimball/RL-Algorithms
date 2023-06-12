import os

import dill
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.distributions import Categorical

matplotlib.use('Agg')
dir_path = os.path.dirname(os.path.realpath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# random seeding
torch.manual_seed(0)
gamma = 0.9
alpha_critic = 3e-3
alpha_actor = 1e-3
episodes = 200
num_runs = 3
env = gym.make('LunarLander-v2')


class Actor(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.Mish(),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.mlp(x)


class Critic(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(8, 32),
            nn.Mish(),
            nn.Linear(32, 32),
            nn.Mish(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.mlp(x)


runs = []
tde_over_runs = []
for run in tqdm(range(num_runs)):
    actor = Actor().to(device)
    critic = Critic().to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=alpha_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=alpha_critic)

    # initialization
    obs, info = env.reset(seed=run)
    returns = []
    tdes = []

    for eps in range(episodes):
        obs = torch.from_numpy(obs).to(device)
        terminated = False
        truncated = False
        I = 1.0
        eps_return = 0.0
        total_tde = 0.0
        while not terminated and not truncated:
            logits = actor(obs)
            policy = Categorical(logits=logits)
            a = policy.sample()

            next_obs, reward, terminated, truncated, info = env.step(a.item())

            next_obs = torch.from_numpy(next_obs).to(device)

            eps_return += reward*I

            if terminated:
                G = 0.0
            else:
                G = reward + gamma*critic(next_obs).item()

            tde = G - critic(obs)

            critic_loss = torch.square(tde)
            actor_loss = -policy.log_prob(a) * tde.detach()*I

            critic_loss.backward()
            actor_loss.backward()
            critic_opt.step()
            actor_opt.step()
            critic_opt.zero_grad()
            actor_opt.zero_grad()

            total_tde += abs(tde.item())
            obs = next_obs
            I = gamma * I
        obs, info = env.reset()
        returns.append(eps_return)
        tdes.append(total_tde)

    runs.append(returns)
    tde_over_runs.append(tdes)
    # with open(os.path.join(dir_path, 'weights', f'critic_params_{run}.pickle'), 'wb') as f:
    #     dill.dump(critic_params, f)
    # with open(os.path.join(dir_path, 'weights', f'actor_params_{run}.pickle'), 'wb') as f:
    #     dill.dump(actor_params, f)

env.close()

runs = np.array(runs)
runs_mean = np.mean(runs, axis=0)
runs_std = np.std(runs, axis=0)

fig = plt.figure()
sns.lineplot(x=np.arange(episodes), y=runs_mean)
upper_bound = runs_mean + runs_std
lower_bound = runs_mean - runs_std
plt.fill_between(np.arange(episodes), upper_bound, lower_bound, alpha=0.3)
plt.title('Actor-Critic Return')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.savefig(os.path.join(dir_path, 'Actor_Critic_Return.png'))
plt.close(fig)

tdes_over_runs = np.array(tde_over_runs)
tdes_mean = np.mean(tdes_over_runs, axis=0)
tdes_std = np.std(tdes_over_runs, axis=0)

plt.figure()
sns.lineplot(x=np.arange(episodes), y=tdes_mean)
upper_bound = tdes_mean + tdes_std
lower_bound = tdes_mean - tdes_std
plt.fill_between(np.arange(episodes), upper_bound, lower_bound, alpha=0.3)
plt.title('Actor-Critic TD Error')
plt.xlabel('Episode')
plt.ylabel('Sum of TD Error')
plt.savefig(os.path.join(dir_path, 'Actor_Critic_TD_Error.png'))
