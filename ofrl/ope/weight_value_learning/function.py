"""Weight and Value Functions."""
import torch
from torch import nn
import torch.nn.functional as F


class VFunction(nn.Module):
    """Value Function (for both discrete and continuous action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.VFunction`

    Parameters
    -------
    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.
    
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        return self.fc2(x).flatten()


class StateWeightFunction(nn.Module):
    """State Weight Function (for both discrete and continuous action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.StateWeightFunction`

    Parameters
    -------
    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.
    
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        return F.relu(self.fc2(x)).flatten()


class DiscreteQFunction(nn.Module):
    """Q Function (for discrete action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.DiscreteQFunction`

    Parameters
    -------
    n_actions: int (> 0)
        Number of discrete actions.

    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.

    device: str, default="cuda:0"
        Specifies device used for torch.
    
    """
    def __init__(
        self,
        n_actions: int,
        state_dim: int,
        hidden_dim: int = 100,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.action_onehot = torch.eye(n_actions, device=device)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        action = self.action_onehot[action]
        x = F.relu(self.fc1(state))
        values = self.fc2(x)
        return (values * action).sum(axis=1)

    def all(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

    def max(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        values = self.fc2(x)
        return torch.max(values, dim=1)[0]

    def argmax(
        self,
        state: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        values = self.fc2(x)
        return torch.max(values, dim=1)[1]

    def expectation(
        self,
        state: torch.Tensor,
        action_distribution: torch.Tensor,
    ):
        x = F.relu(self.fc1(state))
        values = self.fc2(x)
        return (values * action_distribution).sum(axis=1)


class ContinuousQFunction(nn.Module):
    """Q Function (for continuous action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.ContinuousQFunction`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of the actions.

    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.
    
    """
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        values = self.fc2(x)
        return (values * action).sum(axis=1)


class DiscreteStateActionWeightFunction(nn.Module):
    """State Action Weight Function (for discrete action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.DiscreteStateActionWeightFunction`

    Parameters
    -------
    n_actions: int (> 0)
        Number of discrete actions.

    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.

    device: str, default="cuda:0"
        Specifies device used for torch.
    
    """
    def __init__(
        self,
        n_actions: int,
        state_dim: int,
        hidden_dim: int = 100,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.action_onehot = torch.eye(n_actions, device=device)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        action = self.action_onehot[action]
        x = F.relu(self.fc1(state))
        values = F.relu(self.fc2(x))
        return (values * action).sum(axis=1)


class ContinuousStateActionWeightFunction(nn.Module):
    """State Action Weight Function (for continuous action space).
    
    Bases: :class:`torch.nn.Module`

    Imported as: :class:`ofrl.ope.weight_value_learning.function.ContinuousStateActionWeightFunction`

    Parameters
    -------
    action_dim: int (> 0)
        Dimensions of the actions.

    state_dim: int (> 0)
        Dimensions of the states.

    hidden_dim: int, default=100 (> 0)
        Hidden dimension of the network.
    
    """
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 100,
    ):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))