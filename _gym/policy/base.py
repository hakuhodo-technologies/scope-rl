"""Abstract Base Class for Policy."""
from abc import ABCMeta, abstractmethod
from typing import Union
from typing import Tuple
from nptyping import NDArray

# fix later
class BasePolicy(metaclass=ABCMeta):
    "Base class for reinforcement learning (RL) agent (i.e., policy)"
    
    @abstractmethod
    def act(self, state: NDArray[float]) -> Tuple[Union[int, float]]:
        """Choose action to take."""
        raise NotImplementedError
        
    @abstractmethod
    def update(self):
        """Update the policy."""
        raise NotImplementedError