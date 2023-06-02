# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

import gym
from sklearn.linear_model import LogisticRegression

from rtbgym.envs.rtb import RTBEnv
from rtbgym.envs.wrapper_rtb import CustomizedRTBEnv
from rtbgym.envs.simulator.function import (
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)

from .version import __version__  # noqa


__all__ = [
    "RTBEnv",
    "CustomizedRTBEnv",
    "BaseWinningPriceDistribution",
    "BaseClickAndConversionRate",
]


# register standard environment
env = RTBEnv(random_state=12345)
# discrete environment
gym.envs.register(
    id="RTBEnv-discrete-v0",
    entry_point="rtbgym.envs.wrapper_rtb:CustomizedRTBEnv",
    nondeterministic=True,
    kwargs={
        "original_env": env,
        "reward_predictor": LogisticRegression(),
        "action_type": "discrete",
    },
)
# continuous environment
gym.envs.register(
    id="RTBEnv-continuous-v0",
    entry_point="rtbgym.envs.wrapper_rtb:CustomizedRTBEnv",
    nondeterministic=True,
    kwargs={
        "original_env": env,
        "reward_predictor": LogisticRegression(),
        "action_type": "continuous",
    },
)
