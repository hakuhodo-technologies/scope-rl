# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

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
            Number of trajectories to generate by rolling out the behavior policy.

        Returns
        -------
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            :class:`MultipleLoggedDataset` is an instance containing (multiple) logged datasets.

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
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension in the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed under the behavior policy.

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
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).


        """
        raise NotImplementedError

    @abstractmethod
    def obtain_steps(self, n_trajectories: int) -> LoggedDataset:
        """Rollout behavior policy and obtain steps.

        Parameters
        -------
        n_trajectories: int, default=10000 (> 0)
            Number of trajectories to generate by rolling out the behavior policy.

        Returns
        -------
        logged_dataset(s): LoggedDataset or MultipleLoggedDataset
            :class:`MultipleLoggedDataset` is an instance containing (multiple) logged datasets.

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
                Type of the action space.
                Either "discrete" or "continuous".

            n_actions: int (> 0)
                Number of actions.
                If action_type is "continuous", `None` is recorded.

            action_dim: int (> 0)
                Dimensions of the action space.
                If action_type is "discrete", `None` is recorded.

            action_keys: list of str
                Name of each dimension of the action space.
                If action_type is "discrete", `None` is recorded.

            action_meaning: dict
                Dictionary to map discrete action index to a specific action.
                If action_type is "continuous", `None` is recorded.

            state_dim: int (> 0)
                Dimensions of the state space.

            state_keys: list of str
                Name of each dimension of the state space.

            state: ndarray of shape (size, state_dim)
                State observed under the behavior policy.

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
                Propensity of the observed action being chosen under the behavior policy (pscore stands for propensity score).

        """
        raise NotImplementedError
