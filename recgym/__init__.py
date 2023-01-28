import gym

from recgym.envs.rec import RECEnv
from recgym.envs.simulator.function import UserModel
from recgym.envs.simulator.base import BaseUserModel
from recgym.types import Action

__all__ = ["RECEnv", "UserModel", "BaseUserModel", "Action"]

# register standard environment
env = RECEnv(random_state=12345)

# environment
gym.envs.register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
)
