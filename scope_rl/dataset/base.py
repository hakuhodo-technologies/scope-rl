"""Abstract base class for logged dataset."""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from ..types import LoggedDataset


@dataclass
class BaseDataset(metaclass=ABCMeta):
    """Base class for logged dataset.

    Imported as: :class:`scope_rl.dataset.BaseDataset`

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
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            :class:`MultipleLoggedDataset` is a instance containing (multiple) logged datasets.

            For API consistency, each logged dataset should contain the following.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Action choice probability of the behavior policy for the chosen action.


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
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            :class:`MultipleLoggedDataset` is a instance containing (multiple) logged datasets.

            For API consistency, each logged dataset should contain the following.

            .. code-block:: python

                key: [
                    size,
                    n_trajectories,
                    step_per_trajectory,
                    action_type,
                    n_actions,
                    action_dim,
                    action_keys,
                    action_meaning,
                    state_dim,
                    state_keys,
                    state,
                    action,
                    reward,
                    done,
                    terminal,
                    info,
                    pscore,
                ]

            size: int (> 0)
                Number of steps the dataset records.

            n_trajectories: int (> 0)
                Number of trajectories the dataset records.

            step_per_trajectory: int (> 0)
                Number of timesteps in an trajectory.

            action_type: str
                Action type of the RL agent.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of discrete actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the actions.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of the action variable at each dimension.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary which maps discrete action index into specific actions.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the states.

            state_keys: list of str
                Name of the state variable at each dimension.

            state: ndarray of shape (size, state_dim)
                State observed by the behavior policy.

            action: ndarray of shape (size, ) or (size, action_dim)
                Action chosen by the behavior policy.

            reward: ndarray of shape (size, )
                Reward observed for each (state, action) pair.

            done: ndarray of shape (size, )
                Whether an episode ends or not.

            terminal: ndarray of shape (size, )
                Whether an episode reaches the pre-defined maximum steps.

            info: dict
                Additional feedbacks from the environment.

            pscore: ndarray of shape (size, )
                Action choice probability of the behavior policy for the chosen action.

        """
        raise NotImplementedError
