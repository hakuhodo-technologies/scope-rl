# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

import gym

from basicgym.envs.synthetic import BasicEnv
from basicgym.envs.simulator.base import BaseStateTransitionFunction
from basicgym.envs.simulator.base import BaseRewardFunction

from .version import __version__  # noqa


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
