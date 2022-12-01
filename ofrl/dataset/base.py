"""Abstract Base class for Logged Dataset."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from ..types import LoggedDataset


@dataclass
class BaseDataset(metaclass=ABCMeta):
    """Base class for logged dataset."""

    @abstractmethod
    def obtain_episodes(self, n_trajectories: int) -> LoggedDataset:
        """Rollout behavior policy and obtain trajectories."""
        raise NotImplementedError

    @abstractmethod
    def obtain_steps(self, n_trajectories: int) -> LoggedDataset:
        """Rollout behavior policy and obtain steps."""
        raise NotImplementedError
