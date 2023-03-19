import gym

from recgym.envs.rec import RECEnv
from recgym.envs.simulator.base import BaseUserModel

__all__ = ["RECEnv", "BaseUserModel"]

# register standard environment
gym.envs.register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
    nondeterministic=True,
    kwargs={
        "random_state": 12345,
    },
)
