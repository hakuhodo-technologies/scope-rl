from ofrl.ope.ops import OffPolicySelection
from ofrl.ope.input import CreateOPEInput
from ofrl.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseStateMarginalOffPolicyEstimator,
    BaseStateActionMarginalOffPolicyEstimator,
    BaseCumulativeDistributionOffPolicyEstimator,
)
from ofrl.ope.ope import (
    OffPolicyEvaluation,
    CumulativeDistributionOffPolicyEvaluation,
)
from ofrl.ope.basic_estimators_discrete import (
    DiscreteDirectMethod,
    DiscreteTrajectoryWiseImportanceSampling,
    DiscretePerDecisionImportanceSampling,
    DiscreteDoublyRobust,
    DiscreteSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteSelfNormalizedPerDecisionImportanceSampling,
    DiscreteSelfNormalizedDoublyRobust,
)
from ofrl.ope.marginal_estimators_discrete import (
    DiscreteStateMarginalImportanceSampling,
    DiscreteStateMarginalDoublyRobust,
    DiscreteStateMarginalSelfNormalizedImportanceSampling,
    DiscreteStateActionMarginalSelfNormalizedDoublyRobust,
    DiscreteStateActionMarginalImportanceSampling,
    DiscreteStateActionMarginalDoublyRobust,
    DiscreteStateActionMarginalSelfNormalizedImportanceSampling,
    DiscreteStateMarginalSelfNormalizedDoublyRobust,
)
from ofrl.ope.cumulative_distribution_estimators_discrete import (
    DiscreteCumulativeDistributionDirectMethod,
    DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust,
)
from ofrl.ope.weight_value_learning.augemented_lagrangian_dual_learning_discrete import (
    DiscreteAugmentedLagrangianStateActionWightValueLearning,
    DiscreteAugmentedLagrangianStateWightValueLearning,
)
from ofrl.ope.weight_value_learning.minimax_value_learning_discrete import (
    DiscreteMinimaxStateActionValueLearning,
    DiscreteMinimaxStateValueLearning,
)
from ofrl.ope.weight_value_learning.minimax_weight_learning_discrete import (
    DiscreteMinimaxStateActionWeightLearning,
    DiscreteMinimaxStateWeightLearning,
)
from ofrl.ope.basic_estimators_continuous import (
    ContinuousDirectMethod,
    ContinuousTrajectoryWiseImportanceSampling,
    ContinuousPerDecisionImportanceSampling,
    ContinuousDoublyRobust,
    ContinuousSelfNormalizedTrajectoryWiseImportanceSampling,
    ContinuousSelfNormalizedPerDecisionImportanceSampling,
    ContinuousSelfNormalizedDoublyRobust,
)
from ofrl.ope.marginal_estimators_continuous import (
    ContinuousStateMarginalImportanceSampling,
    ContinuousStateMarginalDoublyRobust,
    ContinuousStateMarginalSelfNormalizedImportanceSampling,
    ContinuousStateActionMarginalSelfNormalizedDoublyRobust,
    ContinuousStateActionMarginalImportanceSampling,
    ContinuousStateActionMarginalDoublyRobust,
    ContinuousStateActionMarginalSelfNormalizedImportanceSampling,
    ContinuousStateMarginalSelfNormalizedDoublyRobust,
)
from ofrl.ope.cumulative_distribution_estimators_continuous import (
    ContinuousCumulativeDistributionDirectMethod,
    ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling,
    ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust,
    ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling,
    ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust,
)
from ofrl.ope.weight_value_learning.augemented_lagrangian_dual_learning_continuous import (
    ContinuousAugmentedLagrangianStateActionWightValueLearning,
    ContinuousAugmentedLagrangianStateWightValueLearning,
)
from ofrl.ope.weight_value_learning.minimax_value_learning_continuous import (
    ContinuousMinimaxStateActionValueLearning,
    ContinuousMinimaxStateValueLearning,
)
from ofrl.ope.weight_value_learning.minimax_weight_learning_contiuous import (
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
    "DiscreteStateMarginalImportanceSampling",
    "DiscreteStateMarginalDoublyRobust",
    "DiscreteStateMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateMarginalSelfNormalizedDoublyRobust",
    "DiscreteStateActionMarginalImportanceSampling",
    "DiscreteStateActionMarginalDoublyRobust",
    "DiscreteStateActionMarginalSelfNormalizedImportanceSampling",
    "DiscreteStateActionMarginalSelfNormalizedDoublyRobust",
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
