# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

import copy
from typing import ClassVar, Sequence, Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from d3rlpy.models.torch import Encoder, EncoderWithAction
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.encoders import _create_activation


class StateEncoder(nn.Module):
    def __init__(self, n_unique_states: int, dim_emb: int = 5):
        super().__init__()
        self.embedding = nn.Embedding(n_unique_states, dim_emb)
        self.n_unique_state = n_unique_states
        self.dim_emb = dim_emb

    def forward(self, state_ids: torch.Tensor):
        return torch.cat(
            [self.embedding(state_ids[:, i]) for i in range(state_ids.shape[1])], dim=1
        )

    def encode(self, state_ids: np.ndarray, device: str = "cuda:0"):
        state_ids = torch.LongTensor(state_ids, device=device)
        with torch.no_grad():
            emb = self(state_ids).cpu().detach().numpy()
        return emb


class _EmbeddingEncoder(nn.Module):  # type: ignore
    _observation_shape: Sequence[int]
    _n_unique_states: int
    _dim_emb: int
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool
    _activation: nn.Module
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        n_unique_states: int,
        dim_emb: int = 5,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self._observation_shape = observation_shape

        if hidden_units is None:
            hidden_units = [256, 256]

        self._n_unique_states = n_unique_states
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._feature_size = hidden_units[-1]
        self._dim_emb = dim_emb
        self._activation = activation
        self._use_dense = use_dense

        in_units = [observation_shape[0] * dim_emb] + list(hidden_units[:-1])
        self._encoder = StateEncoder(
            n_unique_states=n_unique_states,
            dim_emb=dim_emb,
        )
        self._fcs = nn.ModuleList()
        self._bns = nn.ModuleList()
        self._dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(in_units, hidden_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0] * dim_emb
            self._fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._dropouts.append(nn.Dropout(dropout_rate))

    def _fc_encode(self, x: torch.Tensor) -> torch.Tensor:
        h = x = self._encoder(x.long())
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    def get_feature_size(self) -> int:
        return self._feature_size

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    @property
    def n_unique_states(self) -> int:
        return self._n_unique_states

    @property
    def dim_emb(self) -> int:
        return self._dim_emb

    @property
    def last_layer(self) -> nn.Linear:
        return self._fcs[-1]

    @property
    def state_encoder(self) -> nn.Module:
        return self._encoder


class EmbeddingEncoder(_EmbeddingEncoder, Encoder):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h


class EmbeddingEncoderWithAction(_EmbeddingEncoder, EncoderWithAction):
    _action_size: int
    _discrete_action: bool

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        n_unique_states: int,
        dim_emb: int = 5,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        discrete_action: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        self._action_size = action_size
        self._discrete_action = discrete_action
        concat_shape = (observation_shape[0] + action_size,)
        super().__init__(
            observation_shape=concat_shape,
            n_unique_states=n_unique_states,
            dim_emb=dim_emb,
            hidden_units=hidden_units,
            use_batch_norm=use_batch_norm,
            use_dense=use_dense,
            dropout_rate=dropout_rate,
            activation=activation,
        )
        self._observation_shape = observation_shape

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self._discrete_action:
            action = F.one_hot(
                action.view(-1).long(), num_classes=self.action_size
            ).float()
        h = self._fc_encode(x, action)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        return h

    def _fc_encode(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = x = self._encoder(x.long())
        h = torch.cat([h, action], dim=1)
        for i, fc in enumerate(self._fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._bns[i](h)
            if self._dropout_rate is not None:
                h = self._dropouts[i](h)
        return h

    @property
    def action_size(self) -> int:
        return self._action_size


class EmbeddingEncoderFactory(EncoderFactory):
    """Embedding encoder factory class.

    This is a customized version of the vector encoder factory in d3rlpy.

    """

    TYPE: ClassVar[str] = "vector"
    _n_unique_states: int
    _dim_emb: int
    _hidden_units: Sequence[int]
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool

    def __init__(
        self,
        n_unique_states: int,
        dim_emb: int = 5,
        hidden_units: Optional[Sequence[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
    ):
        if hidden_units is None:
            self._hidden_units = [256, 256]
        else:
            self._hidden_units = hidden_units
        self._n_unique_states = n_unique_states
        self._dim_emb = dim_emb
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._use_dense = use_dense

    def create(self, observation_shape: Sequence[int]) -> EmbeddingEncoder:
        assert len(observation_shape) == 1
        return EmbeddingEncoder(
            observation_shape=observation_shape,
            n_unique_states=self._n_unique_states,
            dim_emb=self._dim_emb,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        discrete_action: bool = False,
    ) -> EmbeddingEncoderWithAction:
        assert len(observation_shape) == 1
        return EmbeddingEncoderWithAction(
            observation_shape=observation_shape,
            n_unique_states=self._n_unique_states,
            dim_emb=self._dim_emb,
            action_size=action_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            hidden_units = copy.deepcopy(self._hidden_units)
        else:
            hidden_units = self._hidden_units
        params = {
            "hidden_units": hidden_units,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
            "use_dense": self._use_dense,
        }
        return params
