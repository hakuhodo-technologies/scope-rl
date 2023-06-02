# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

import gym

from recgym.envs.rec import RECEnv
from recgym.envs.simulator.base import BaseUserModel

from .version import __version__  # noqa


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
