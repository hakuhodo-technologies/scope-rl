# Copyright (c) 2023, Haruka Kiyohara, Ren Kishimoto, HAKUHODO Technologies Inc., and Hanjuku-kaso Co., Ltd. All rights reserved.
# Licensed under the Apache 2.0 License.

from scope_rl.ope.weight_value_learning.base import BaseWeightValueLearner
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_discrete import (
    DiscreteDiceStateActionWightValueLearning,
    DiscreteDiceStateWightValueLearning,
)
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_continuous import (
    ContinuousDiceStateActionWightValueLearning,
    ContinuousDiceStateWightValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_value_learning_discrete import (
    DiscreteMinimaxStateActionValueLearning,
    DiscreteMinimaxStateValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_value_learning_continuous import (
    ContinuousMinimaxStateActionValueLearning,
    ContinuousMinimaxStateValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_weight_learning_discrete import (
    DiscreteMinimaxStateActionWeightLearning,
    DiscreteMinimaxStateWeightLearning,
)
from scope_rl.ope.weight_value_learning.minimax_weight_learning_continuous import (
    ContinuousMinimaxStateActionWeightLearning,
    ContinuousMinimaxStateWeightLearning,
)


__all__ = [
    "BaseWeightValueLearner",
    "DiscreteDiceStateActionWightValueLearning",
    "DiscreteDiceStateWightValueLearning",
    "ContinuousDiceStateActionWightValueLearning",
    "ContinuousDiceStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]


__discrete__ = [
    "DiscreteDiceStateActionWightValueLearning",
    "DiscreteDiceianStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
]


__continuous__ = [
    "ContinuousDiceStateActionWightValueLearning",
    "ContinuousDiceStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]
