import jax.numpy as jnp
import jax.nn as nn
import haiku as hk


def mish(x: jnp.ndarray):
    return x*jnp.tanh(nn.softplus(x))


def policy_fn(observation):
    mlp = hk.Sequential([
        hk.Linear(32),
        mish,
        hk.Linear(32),
        mish,
        hk.Linear(4)])
    return mlp(observation)


def val_fn(observation):
    mlp = hk.Sequential([
        hk.Linear(32),
        mish,
        hk.Linear(32),
        mish,
        hk.Linear(1)])
    return mlp(observation)
