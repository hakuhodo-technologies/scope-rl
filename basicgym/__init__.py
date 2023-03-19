import gym

from basicgym.envs.synthetic import BasicEnv
from basicgym.envs.simulator.base import BaseStateTransitionFunction
from basicgym.envs.simulator.base import BaseRewardFunction

__all__ = [
    "BasicEnv",
    "BaseStateTransitionFunction",
    "BaseRewardFunction",
]


# register standard environment
# discrete environment
gym.envs.register(
    id="BasicEnv-discrete-v0",
    entry_point="basicgym.envs.synthetic:BasicEnv",
    nondeterministic=True,
    kwargs={
        "action_type": "discrete",
        "random_state": 12345,
    },
)
# continuous environment
gym.envs.register(
    id="BasicEnv-continuous-v0",
    entry_point="basicgym.envs.synthetic:BasicEnv",
    nondeterministic=True,
    kwargs={
        "action_type": "continuous",
        "random_state": 12345,
    },
)
