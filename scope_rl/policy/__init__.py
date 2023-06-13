# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

from scope_rl.policy.opl import TrainCandidatePolicies
from scope_rl.policy.head import (
    BaseHead,
    OnlineHead,
    DiscreteEpsilonGreedyHead,
    DiscreteSoftmaxHead,
    ContinuousGaussianHead,
    ContinuousTruncatedGaussianHead,
    ContinuousEvalHead,
)
from scope_rl.policy.encoder import (
    StateEncoder,
    EmbeddingEncoderFactory,
)


__all__ = [
    "TrainCandidatePolicies",
    "BaseHead",
    "OnlineHead",
    "DiscreteEpsilonGreedyHead",
    "DiscreteSoftmaxHead",
    "ContinuousGaussianHead",
    "ContinuousTruncatedGaussianHead",
    "ContinuousEvalHead",
    "StateEncoder",
    "EmbeddingEncoderFactory",
]


__head__ = [
    "BaseHead",
    "OnlineHead",
    "DiscreteEpsilonGreedyHead",
    "DiscreteSoftmaxHead",
    "ContinuousGaussianHead",
    "ContinuousTruncatedGaussianHead",
    "ContinuousEvalHead",
]


__encoder__ = [
    "StateEncoder",
    "EmbeddingEncoderFactory",
]
