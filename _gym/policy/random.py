from typing import Tuple, Union
from nptyping import NDArray

import gym

from .base import BasePolicy

# fix later
class RandomPolicy(BasePolicy):
    def __init__(self, env: gym.Env):
        self.action_type = env.action_type
        self.action_space = env.action_space
    
    def act(self, state: NDArray[float]) -> Tuple[Union[int, float]]:
        if self.action_type == "discrete":
            return self.action_space.sample(), 1 / self.action_space.n
        else:
            return self.action_space.sample(), 1 / (self.action_space.high - self.action_space.low)
        
    def update(self, *args):
        pass