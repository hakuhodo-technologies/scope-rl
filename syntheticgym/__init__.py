import gym

from syntheticgym.envs.synthetic import SyntheticEnv
from syntheticgym.envs.simulator.function import StateTransition
from syntheticgym.envs.simulator.function import RewardFunction
from syntheticgym.envs.simulator.base import BaseStateTransition
from syntheticgym.envs.simulator.base import BaseRewardFunction
from syntheticgym.types import Action

__all__ = ["SyntheticEnv","StateTransition", "RewardFunction","BaseStateTransition", "BaseRewardFunction",  "Action"]

# environment
gym.envs.register(
    id="SyntheticEnv-v0",
    entry_point="syntheticgym.envs.synthetic:SyntheticEnv",
    kwargs={
        "random_state": 12345,
    },
)
