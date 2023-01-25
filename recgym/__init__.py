import gym

from recgym.envs.rec import RECEnv
from recgym.envs.function import UserModel
from recgym.envs.base import BaseUserModel

__all__ = [
    "RECEnv",
    "UserModel"
    "BaseUserModel"
]

# register standard environment
env = RECEnv(random_state=12345)

# environment
gym.envs.register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
)
