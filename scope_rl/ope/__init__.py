from scope_rl.ope.ops import OffPolicySelection
from scope_rl.ope.input import CreateOPEInput
from scope_rl.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseStateMarginalOPEEstimator,
    BaseStateActionMarginalOPEEstimator,
    BaseCumulativeDistributionOPEEstimator,
)
from scope_rl.ope.ope import (
    OffPolicyEvaluation,
    CumulativeDistributionOPE,
)
from scope_rl.ope.basic_estimators_discrete import (
    DiscreteDirectMethod,
    DiscreteTrajectoryWiseImportanceSampling,
    DiscretePerDecisionImportanceSampling,
    DiscreteDoublyRobust,
    DiscreteSelfNormalizedTIS,
    DiscreteSelfNormalizedPDIS,
    DiscreteSelfNormalizedDR,
)
from scope_rl.ope.marginal_estimators_discrete import (
    DiscreteDoubleReinforcementLearning,
    DiscreteStateMarginalIS,
    DiscreteStateMarginalDR,
    DiscreteStateMarginalSNIS,
    DiscreteStateActionMarginalSNDR,
    DiscreteStateActionMarginalIS,
    DiscreteStateActionMarginalDR,
    DiscreteStateActionMarginalSNIS,
    DiscreteStateMarginalSNDR,
)
from scope_rl.ope.cumulative_distribution_estimators_discrete import (
    DiscreteCumulativeDistributionDM,
    DiscreteCumulativeDistributionTIS,
    DiscreteCumulativeDistributionTDR,
    DiscreteCumulativeDistributionSNTIS,
    DiscreteCumulativeDistributionSNTDR,
)
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_discrete import (
    DiscreteDiceStateActionWightValueLearning,
    DiscreteDiceStateWightValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_value_learning_discrete import (
    DiscreteMinimaxStateActionValueLearning,
    DiscreteMinimaxStateValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_weight_learning_discrete import (
    DiscreteMinimaxStateActionWeightLearning,
    DiscreteMinimaxStateWeightLearning,
)
from scope_rl.ope.basic_estimators_continuous import (
    ContinuousDirectMethod,
    ContinuousTrajectoryWiseImportanceSampling,
    ContinuousPerDecisionImportanceSampling,
    ContinuousDoublyRobust,
    ContinuousSelfNormalizedTIS,
    ContinuousSelfNormalizedPDIS,
    ContinuousSelfNormalizedDR,
)
from scope_rl.ope.marginal_estimators_continuous import (
    ContinuousDoubleReinforcementLearning,
    ContinuousStateMarginalIS,
    ContinuousStateMarginalDR,
    ContinuousStateMarginalSNIS,
    ContinuousStateActionMarginalSNDR,
    ContinuousStateActionMarginalIS,
    ContinuousStateActionMarginalDR,
    ContinuousStateActionMarginalSNIS,
    ContinuousStateMarginalSNDR,
)
from scope_rl.ope.cumulative_distribution_estimators_continuous import (
    ContinuousCumulativeDistributionDM,
    ContinuousCumulativeDistributionTIS,
    ContinuousCumulativeDistributionTDR,
    ContinuousCumulativeDistributionSNTIS,
    ContinuousCumulativeDistributionSNTDR,
)
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_continuous import (
    ContinuousDiceStateActionWightValueLearning,
    ContinuousDiceStateWightValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_value_learning_continuous import (
    ContinuousMinimaxStateActionValueLearning,
    ContinuousMinimaxStateValueLearning,
)
from scope_rl.ope.weight_value_learning.minimax_weight_learning_continuous import (
    ContinuousMinimaxStateActionWeightLearning,
    ContinuousMinimaxStateWeightLearning,
)


__all__ = [
    "BaseOffPolicyEstimator",
    "BaseStateMarginalOPEEstimator",
    "BaseStateActionMarginalOPEEstimator",
    "BaseCumulativeDistributionOPEEstimator",
    "OffPolicyEvaluation",
    "CumulativeDistributionOPE",
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTIS",
    "DiscreteSelfNormalizedPDIS",
    "DiscreteSelfNormalizedDR",
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalIS",
    "DiscreteStateMarginalDR",
    "DiscreteStateMarginalSNIS",
    "DiscreteStateMarginalSNDR",
    "DiscreteStateActionMarginalIS",
    "DiscreteStateActionMarginalDR",
    "DiscreteStateActionMarginalSNIS",
    "DiscreteStateActionMarginalSNDR",
    "DiscreteCumulativeDistributionDM",
    "DiscreteCumulativeDistributionTIS",
    "DiscreteCumulativeDistributionTDR",
    "DiscreteCumulativeDistributionSNTIS",
    "DiscreteCumulativeDistributionSNTDR",
    "DiscreteDiceStateActionWightValueLearning",
    "DiscreteDiceStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousPerDecisionImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTIS",
    "ContinuousSelfNormalizedPDIS",
    "ContinuousSelfNormalizedDR",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalIS",
    "ContinuousStateMarginalDR",
    "ContinuousStateMarginalSNIS",
    "ContinuousStateMarginalSNDR",
    "ContinuousStateActionMarginalIS",
    "ContinuousStateActionMarginalDR",
    "ContinuousStateActionMarginalSNIS",
    "ContinuousStateActionMarginalSNDR",
    "ContinuousCumulativeDistributionDM",
    "ContinuousCumulativeDistributionTIS",
    "ContinuousCumulativeDistributionTDR",
    "ContinuousCumulativeDistributionSNTIS",
    "ContinuousCumulativeDistributionSNTDR",
    "ContinuousDiceStateActionWightValueLearning",
    "ContinuousDiceStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]


__base__ = [
    "BaseOffPolicyEstimator",
    "BaseCumulativeDistributionOPEEstimator",
]


__meta__ = [
    "OffPolicyEvaluation",
    "CumulativeDistributionOPE",
    "OffPolicySelection",
    "CreateOPEInput",
]


__basic__ = [
    "BaseOffPolicyEstimator",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTIS",
    "DiscreteSelfNormalizedPDIS",
    "DiscreteSelfNormalizedDR",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousPerDecisionImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTIS",
    "ContinuousSelfNormalizedPDIS",
    "ContinuousSelfNormalizedDR",
]


__marginal__ = [
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalIS",
    "DiscreteStateMarginalDR",
    "DiscreteStateMarginalSNIS",
    "DiscreteStateMarginalSNDR",
    "DiscreteStateActionMarginalIS",
    "DiscreteStateActionMarginalDR",
    "DiscreteStateActionMarginalSNIS",
    "DiscreteStateActionMarginalSNDR",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalIS",
    "ContinuousStateMarginalDR",
    "ContinuousStateMarginalSNIS",
    "ContinuousStateMarginalSNDR",
    "ContinuousStateActionMarginalIS",
    "ContinuousStateActionMarginalDR",
    "ContinuousStateActionMarginalSNIS",
    "ContinuousStateActionMarginalSNDR",
]


__cumulative__ = [
    "BaseCumulativeDistributionOPEEstimator",
    "DiscreteCumulativeDistributionDM",
    "DiscreteCumulativeDistributionTIS",
    "DiscreteCumulativeDistributionTDR",
    "DiscreteCumulativeDistributionSNTIS",
    "DiscreteCumulativeDistributionSNTDR",
    "ContinuousCumulativeDistributionDM",
    "ContinuousCumulativeDistributionTIS",
    "ContinuousCumulativeDistributionTDR",
    "ContinuousCumulativeDistributionSNTIS",
    "ContinuousCumulativeDistributionSNTDR",
]


__learning__ = [
    "DiscreteDiceStateActionWightValueLearning",
    "DiscreteDiceStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
    "ContinuousDiceStateActionWightValueLearning",
    "ContinuousDiceStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]


__discrete__ = [
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTIS",
    "DiscreteSelfNormalizedPDIS",
    "DiscreteSelfNormalizedDR",
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalIS",
    "DiscreteStateMarginalDR",
    "DiscreteStateMarginalSNIS",
    "DiscreteStateMarginalSNDR",
    "DiscreteStateActionMarginalIS",
    "DiscreteStateActionMarginalDR",
    "DiscreteStateActionMarginalSNIS",
    "DiscreteStateActionMarginalSNDR",
    "DiscreteCumulativeDistributionDM",
    "DiscreteCumulativeDistributionTIS",
    "DiscreteCumulativeDistributionTDR",
    "DiscreteCumulativeDistributionSNTIS",
    "DiscreteCumulativeDistributionSNTDR",
    "DiscreteDiceStateActionWightValueLearning",
    "DiscreteDiceStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
]


__continuous__ = [
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousPerDecisionImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTIS",
    "ContinuousSelfNormalizedPDIS",
    "ContinuousSelfNormalizedDR",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalIS",
    "ContinuousStateMarginalDR",
    "ContinuousStateMarginalSNIS",
    "ContinuousStateMarginalSNDR",
    "ContinuousStateActionMarginalIS",
    "ContinuousStateActionMarginalDR",
    "ContinuousStateActionMarginalSNIS",
    "ContinuousStateActionMarginalSNDR",
    "ContinuousCumulativeDistributionDM",
    "ContinuousCumulativeDistributionTIS",
    "ContinuousCumulativeDistributionTDR",
    "ContinuousCumulativeDistributionSNTIS",
    "ContinuousCumulativeDistributionSNTDR",
    "ContinuousDiceStateActionWightValueLearning",
    "ContinuousDiceStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]
