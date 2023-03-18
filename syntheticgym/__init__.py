import gym

from syntheticgym.envs.synthetic import SyntheticEnv
from syntheticgym.envs.simulator.base import BaseStateTransitionFunction
from syntheticgym.envs.simulator.base import BaseRewardFunction

__all__ = [
    "SyntheticEnv",
    "BaseStateTransitionFunction",
    "BaseRewardFunction",
]


# discrete environment
gym.envs.register(
    id="SyntheticEnv-discrete-v0",
    entry_point="syntheticgym.envs.synthetic:SyntheticEnv",
    nondeterministic=True,
    kwargs={
        "action_type": "discrete",
        "random_state": 12345,
    },
)
# continuous environment
gym.envs.register(
    id="SyntheticEnv-continuous-v0",
    entry_point="syntheticgym.envs.synthetic:SyntheticEnv",
    nondeterministic=True,
    kwargs={
        "action_type": "continuous",
        "random_state": 12345,
    },
)
