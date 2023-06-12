import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_return(runs_mean: np.ndarray,
                runs_std: np.ndarray,
                episodes: int,
                path: str):
    fig = plt.figure()
    sns.lineplot(x=np.arange(episodes), y=runs_mean)
    upper_bound = runs_mean + runs_std
    lower_bound = runs_mean - runs_std
    plt.fill_between(np.arange(episodes), upper_bound, lower_bound, alpha=0.3)
    plt.title('Actor-Critic Return')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.savefig(os.path.join(path, 'Actor_Critic_Return.png'))
    plt.close(fig)


def plot_tde(tdes_mean: np.ndarray,
             tdes_std: np.ndarray,
             episodes: int,
             path: str):
    fig = plt.figure()
    sns.lineplot(x=np.arange(episodes), y=tdes_mean)
    upper_bound = tdes_mean + tdes_std
    lower_bound = tdes_mean - tdes_std
    plt.fill_between(np.arange(episodes), upper_bound, lower_bound, alpha=0.3)
    plt.title('Actor-Critic TD Error')
    plt.xlabel('Episode')
    plt.ylabel('Sum of TD Error')
    plt.savefig(os.path.join(path, 'Actor_Critic_TD_Error.png'))
    plt.close(fig)
