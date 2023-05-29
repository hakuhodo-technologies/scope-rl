from scope_rl.ope.ops import OffPolicySelection
from scope_rl.ope.input import CreateOPEInput
from scope_rl.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseStateMarginalOffPolicyEstimator,
    BaseStateActionMarginalOffPolicyEstimator,
    BaseCumulativeDistributionOffPolicyEstimator,
)
from scope_rl.ope.ope import (
    OffPolicyEvaluation,
    CumulativeDistributionOffPolicyEvaluation,
)
from scope_rl.ope.basic_estimators_discrete import (
    DiscreteDirectMethod,
    DiscreteTrajectoryWiseImportanceSampling,
    DiscretePerDecisionImportanceSampling,
    DiscreteDoublyRobust,
    DiscreteSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteSelfNormalizedPerDecisionImportanceSampling,
    DiscreteSelfNormalizedDoublyRobust,
)
from scope_rl.ope.marginal_estimators_discrete import (
    DiscreteDoubleReinforcementLearning,
    DiscreteStateMarginalImportanceSampling,
    DiscreteStateMarginalDoublyRobust,
    DiscreteStateMarginalSelfNormalizedImportanceSampling,
    DiscreteStateActionMarginalSelfNormalizedDoublyRobust,
    DiscreteStateActionMarginalImportanceSampling,
    DiscreteStateActionMarginalDoublyRobust,
    DiscreteStateActionMarginalSelfNormalizedImportanceSampling,
    DiscreteStateMarginalSelfNormalizedDoublyRobust,
)
from scope_rl.ope.cumulative_distribution_estimators_discrete import (
    DiscreteCumulativeDistributionDirectMethod,
    DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust,
)
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_discrete import (
    DiscreteAugmentedLagrangianStateActionWightValueLearning,
    DiscreteAugmentedLagrangianStateWightValueLearning,
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
    ContinuousSelfNormalizedTrajectoryWiseImportanceSampling,
    ContinuousSelfNormalizedPerDecisionImportanceSampling,
    ContinuousSelfNormalizedDoublyRobust,
)
from scope_rl.ope.marginal_estimators_continuous import (
    ContinuousDoubleReinforcementLearning,
    ContinuousStateMarginalImportanceSampling,
    ContinuousStateMarginalDoublyRobust,
    ContinuousStateMarginalSelfNormalizedImportanceSampling,
    ContinuousStateActionMarginalSelfNormalizedDoublyRobust,
    ContinuousStateActionMarginalImportanceSampling,
    ContinuousStateActionMarginalDoublyRobust,
    ContinuousStateActionMarginalSelfNormalizedImportanceSampling,
    ContinuousStateMarginalSelfNormalizedDoublyRobust,
)
from scope_rl.ope.cumulative_distribution_estimators_continuous import (
    ContinuousCumulativeDistributionDirectMethod,
    ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling,
    ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust,
    ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling,
    ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust,
)
from scope_rl.ope.weight_value_learning.augmented_lagrangian_learning_continuous import (
    ContinuousAugmentedLagrangianStateActionWightValueLearning,
    ContinuousAugmentedLagrangianStateWightValueLearning,
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
    "BaseStateMarginalOffPolicyEstimator",
    "BaseStateActionMarginalOffPolicyEstimator",
    "BaseCumulativeDistributionOffPolicyEstimator",
    "OffPolicyEvaluation",
    "CumulativeDistributionOffPolicyEvaluation",
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedPerDecisionImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalImportanceSampling",
    "DiscreteStateMarginalDoublyRobust",
    "DiscreteStateMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateMarginalSelfNormalizedDoublyRobust",
    "DiscreteStateActionMarginalImportanceSampling",
    "DiscreteStateActionMarginalDoublyRobust",
    "DiscreteStateActionMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateActionMarginalSelfNormalizedDoublyRobust",
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "DiscreteAugmentedLagrangianStateActionWightValueLearning",
    "DiscreteAugmentedLagrangianStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousPerDecisionImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousSelfNormalizedPerDecisionImportanceSampling",
    "ContinuousSelfNormalizedDoublyRobust",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalImportanceSampling",
    "ContinuousStateMarginalDoublyRobust",
    "ContinuousStateMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateMarginalSelfNormalizedDoublyRobust",
    "ContinuousStateActionMarginalImportanceSampling",
    "ContinuousStateActionMarginalDoublyRobust",
    "ContinuousStateActionMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateActionMarginalSelfNormalizedDoublyRobust",
    "ContinuousCumulativeDistributionDirectMethod",
    "ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "ContinuousAugmentedLagrangianStateActionWightValueLearning",
    "ContinuousAugmentedLagrangianStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]


__base__ = [
    "BaseOffPolicyEstimator",
    "BaseCumulativeDistributionOffPolicyEstimator",
]


__meta__ = [
    "OffPolicyEvaluation",
    "CumulativeDistributionOffPolicyEvaluation",
    "OffPolicySelection",
    "CreateOPEInput",
]


__basic__ = [
    "BaseOffPolicyEstimator",
    "DiscreteOffPolicyEvaluation",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscreteStepWiseImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedStepWiseImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "ContinuousOffPolicyEvaluation",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousStepWiseImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousSelfNormalizedStepWiseImportanceSampling",
    "ContinuousSelfNormalizedDoublyRobust",
]


__marginal__ = [
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalImportanceSampling",
    "DiscreteStateMarginalDoublyRobust",
    "DiscreteStateMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateMarginalSelfNormalizedDoublyRobust",
    "DiscreteStateActionMarginalImportanceSampling",
    "DiscreteStateActionMarginalDoublyRobust",
    "DiscreteStateActionMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateActionMarginalSelfNormalizedDoublyRobust",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalImportanceSampling",
    "ContinuousStateMarginalDoublyRobust",
    "ContinuousStateMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateMarginalSelfNormalizedDoublyRobust",
    "ContinuousStateActionMarginalImportanceSampling",
    "ContinuousStateActionMarginalDoublyRobust",
    "ContinuousStateActionMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateActionMarginalSelfNormalizedDoublyRobust",
]


__cumulative__ = [
    "BaseCumulativeDistributionOffPolicyEstimator",
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "ContinuousCumulativeDistributionDirectMethod",
    "ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
]


__learning__ = [
    "DiscreteAugmentedLagrangianStateActionWightValueLearning",
    "DiscreteAugmentedLagrangianStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
    "ContinuousAugmentedLagrangianStateActionWightValueLearning",
    "ContinuousAugmentedLagrangianStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]


__discrete__ = [
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionOffPolicyEvaluation",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscreteStepWiseImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedStepWiseImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "DiscreteDoubleReinforcementLearning",
    "DiscreteStateMarginalImportanceSampling",
    "DiscreteStateMarginalDoublyRobust",
    "DiscreteStateMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateMarginalSelfNormalizedDoublyRobust",
    "DiscreteStateActionMarginalImportanceSampling",
    "DiscreteStateActionMarginalDoublyRobust",
    "DiscreteStateActionMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateActionMarginalSelfNormalizedDoublyRobust",
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "DiscreteAugmentedLagrangianStateActionWightValueLearning",
    "DiscreteAugmentedLagrangianStateWightValueLearning",
    "DiscreteMinimaxStateActionValueLearning",
    "DiscreteMinimaxStateValueLearning",
    "DiscreteMinimaxStateActionWeightLearning",
    "DiscreteMinimaxStateWeightLearning",
]


__continuous__ = [
    "ContinuousOffPolicyEvaluation",
    "ContinuousCumulativeDistributionOffPolicyEvaluation",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousStepWiseImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousSelfNormalizedStepWiseImportanceSampling",
    "ContinuousSelfNormalizedDoublyRobust",
    "ContinuousDoubleReinforcementLearning",
    "ContinuousStateMarginalImportanceSampling",
    "ContinuousStateMarginalDoublyRobust",
    "ContinuousStateMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateMarginalSelfNormalizedDoublyRobust",
    "ContinuousStateActionMarginalImportanceSampling",
    "ContinuousStateActionMarginalDoublyRobust",
    "ContinuousStateActionMarginalSelfNormalizedImportanceSampling",
    "ContinuousStateActionMarginalSelfNormalizedDoublyRobust",
    "ContinuousCumulativeDistributionDirectMethod",
    "ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "ContinuousAugmentedLagrangianStateActionWightValueLearning",
    "ContinuousAugmentedLagrangianStateWightValueLearning",
    "ContinuousMinimaxStateActionValueLearning",
    "ContinuousMinimaxStateValueLearning",
    "ContinuousMinimaxStateActionWeightLearning",
    "ContinuousMinimaxStateWeightLearning",
]
