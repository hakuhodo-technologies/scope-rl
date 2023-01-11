import gym
from sklearn.linear_model import LogisticRegression

from rtbgym.envs.rtb import RTBEnv
from rtbgym.envs.wrapper_rtb import CustomizedRTBEnv
from rtbgym.envs.simulator.function import (
    BaseWinningPriceDistribution,
    BaseClickAndConversionRate,
)
from recgym.env.function import inner_reward_function
from recgym.env.function import user_preference_dynamics
from recgym.env.rec import RECEnv


__all__ = [
    "RECEnv",
    "inner_reward_function",
    "user_preference_dynamics",
]

# register standard environment
env = RECEnv(
    reward_function = inner_reward_function,
    state_transition_function = user_preference_dynamics,
)

# environment
gym.envs.register(
    id="RECEnv-v0",
    entry_point="recgym.envs.rec:RECEnv",
)