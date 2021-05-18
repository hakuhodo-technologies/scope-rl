"""Abstract Base class for Logged Dataset."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from _gym.types import LoggedDataset


@dataclass
class BaseDataset(metaclass=ABCMeta):
    """Base class for dataset."""

    @abstractmethod
    def obtain_trajectories(self, n_episodes: int) -> LoggedDataset:
        """Rollout behavior policy and obtain trajectories."""
        raise NotImplementedError

    @abstractmethod
    def calc_on_policy_policy_value(self, n_episodes: int) -> float:
        """Calculate ground-truth policy value of behavior policy by rollout."""
        raise NotImplementedError
