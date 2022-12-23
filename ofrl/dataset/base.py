"""Abstract base class for logged dataset."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from ..types import LoggedDataset


@dataclass
class BaseDataset(metaclass=ABCMeta):
    """Base class for logged dataset.

    Imported as: :class:`ofrl.dataset.BaseDataset`
    
    """

    @abstractmethod
    def obtain_episodes(self, n_trajectories: int) -> LoggedDataset:
        """Rollout behavior policy and obtain episodes.
        
        Parameters
        -------
        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        Returns
        -------
        logged_dataset: LoggedDataset
            Logged dataset.
        
        """
        raise NotImplementedError

    @abstractmethod
    def obtain_steps(self, n_trajectories: int) -> LoggedDataset:
        """Rollout behavior policy and obtain steps.

        Parameters
        -------
        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to rollout the behavior policy and collect data.

        Returns
        -------
        logged_dataset: LoggedDataset
            Logged dataset.
        
        """
        raise NotImplementedError
