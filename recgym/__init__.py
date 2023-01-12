import gym
from sklearn.linear_model import LogisticRegression

from recgym.envs.rec import RECEnv
from recgym.envs.function import inner_reward_function
from recgym.envs.function import user_preference_dynamics


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