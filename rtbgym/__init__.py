import gym

from rtbgym.envs.rtb import RTBEnv
from rtbgym.envs.wrapper_rtb import CustomizedRTBEnv
from rtbgym.envs.simulator.function import (
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)


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
    max_episode_steps=7,
    kwargs={"original_env": env, "action_type": "discrete"},
)
# continuous environment
gym.envs.register(
    id="RTBEnv-continuous-v0",
    entry_point="rtbgym.envs.wrapper_rtb:CustomizedRTBEnv",
    max_episode_steps=7,
    kwargs={"original_env": env, "action_type": "continuous"},
)
