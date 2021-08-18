from typing import Tuple, Union

import gym
import numpy as np

from .base import BasePolicy


# fix later
class RandomPolicy(BasePolicy):
    def __init__(self, env: gym.Env):
        self.action_type = env.action_type
        self.action_space = env.action_space

    def act(self, state: np.ndarray) -> Tuple[Union[int, float]]:
        if self.action_type == "discrete":
            return self.action_space.sample(), 1 / self.action_space.n
        else:
            action = self.action_space.sample()
            action_prob = np.exp(-(action - self.action_space.low))
            return action[0], action_prob[0]

    def update(self, *args):
        pass
