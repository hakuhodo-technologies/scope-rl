from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union, Any

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from _gym.policy import BasePolicy
from _gym.utils import check_logged_dataset


class NeuralEstimator(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


@dataclass
class QFunction(BaseEstimator):
    base_estimator: Union[BaseEstimator, nn.Module]
    optimizer: Optional[torch.optim.Optimizer] = None
    is_continuous: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 12345

    def __post_init__(self):
        self.is_neural_estimator = isinstance(self.base_estimator, nn.Module)

        if self.is_neural_estimator:
            torch.random.manual_seed(self.random_state)

            self.device = torch.device(self.device)
            self.loss_fn = nn.MSELoss()

            if self.optimizer is None:
                self.optimizer = torch.optim.SGD(
                    self.base_estimator.parameters(), lr=1e-3
                )

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
    ):
        if self.is_neural_estimator:
            states = torch.from_numpy(states).float()

            if self.is_continuous:
                actions = torch.from_numpy(actions).float()
                inputs = torch.cat((states, actions), dim=1).to(self.device)
                preds = self.base_estimator(inputs)

            else:
                actions = torch.from_numpy(actions.reshape((-1, 1))).to(torch.int64)
                inputs = states.to(self.device)
                preds = self.base_estimator(inputs)
                preds = torch.gather(preds, 1, actions)

            targets = torch.from_numpy(targets.reshape((-1, 1))).float().to(self.device)
            loss = self.loss_fn(preds, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        else:
            inputs = np.concatenate([states, actions], axis=1)
            self.base_estimator.fit(inputs, targets)

    def predict(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ):
        if self.is_neural_estimator:
            states = torch.from_numpy(states).float()

            with torch.no_grad():
                if self.is_continuous:
                    actions = torch.from_numpy(actions).float()
                    inputs = torch.cat((states, actions), dim=1).to(self.device)
                    preds = self.base_estimator(inputs)

                else:
                    actions = torch.from_numpy(actions.reshape((-1, 1))).to(torch.int64)
                    inputs = states.to(self.device)
                    preds = self.base_estimator(inputs)
                    preds = torch.gather(preds, 1, actions)

                return np.array(preds).flatten()


@dataclass
class FittedQEvaluation:
    base_estimator: Union[BaseEstimator, nn.Module]
    optimizer: Optional[torch.optim.Optimizer] = None
    is_continuous: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_state: int = 12345

    def __post_init__(self):
        self.q_function = QFunction(
            base_estimator=self.base_estimator,
            optimizer=self.optimizer,
            is_continuous=self.is_continuous,
            device=self.device,
            random_state=self.random_state,
        )
        self.random_ = check_random_state(self.random_state)

    def fit(
        self,
        evaluation_policy: BasePolicy,
        logged_dataset: Dict[str, Any],
        batch_size: int = 10000,
        n_iterations: int = 100,
        gamma: float = 1.0,
    ):
        # check_logged_dataset(logged_dataset)
        # assert is_continuous
        data_size = len(logged_dataset["state"])

        for i in tqdm(
            range(n_iterations),
            desc="[Fitted Q Evaluation]",
            total=n_iterations,
        ):
            bootstrap_idx = self.random_.randint(0, data_size, batch_size)
            states = logged_dataset["state"][bootstrap_idx]
            actions = logged_dataset["action"][bootstrap_idx]
            rewards = logged_dataset["reward"][bootstrap_idx]
            next_states = logged_dataset["state"][(bootstrap_idx + 1) % data_size]
            dones = logged_dataset["done"][bootstrap_idx]

            if i == 0:
                next_values = np.empty(batch_size)
            else:
                if self.is_continuous:
                    next_actions = np.empty((batch_size, actions.shape[1]))
                else:
                    next_actions = np.empty(batch_size)
                for i in range(batch_size):
                    next_actions[i] = evaluation_policy.act(next_states[i])[0]
                next_values = self.predict_state_action_value(
                    states=next_states, actions=next_actions
                ) * (1 - dones)

            targets = rewards + gamma * next_values
            self.q_function.update(states=states, actions=actions, targets=targets)

    def predict_state_action_value(
        self,
        states: np.ndarray,
        actions: np.ndarray,
    ):
        return self.q_function.predict(states, actions)
