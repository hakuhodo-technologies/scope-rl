# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

from scope_rl.policy.orl import TrainCandidatePolicies
from scope_rl.policy.head import (
    BaseHead,
    OnlineHead,
    EpsilonGreedyHead,
    SoftmaxHead,
    GaussianHead,
    TruncatedGaussianHead,
    ContinuousEvalHead,
)


__all__ = [
    "TrainCandidatePolicies",
    "BaseHead",
    "OnlineHead",
    "EpsilonGreedyHead",
    "SoftmaxHead",
    "GaussianHead",
    "TruncatedGaussianHead",
    "ContinuousEvalHead",
]


__head__ = [
    "BaseHead",
    "OnlineHead",
    "EpsilonGreedyHead",
    "SoftmaxHead",
    "GaussianHead",
    "TruncatedGaussianHead",
    "ContinuousEvalHead",
]
