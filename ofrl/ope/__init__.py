from ofrl.ope.ops import OffPolicySelection
from ofrl.ope.input import CreateOPEInput
from ofrl.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseCumulativeDistributionOffPolicyEstimator,
)
from ofrl.ope.ope_discrete import (
    DiscreteOffPolicyEvaluation,
    DiscreteCumulativeDistributionOffPolicyEvaluation,
)
from ofrl.ope.estimators_discrete import (
    DiscreteDirectMethod,
    DiscreteTrajectoryWiseImportanceSampling,
    DiscretePerDecisionImportanceSampling,
    DiscreteDoublyRobust,
    DiscreteSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteSelfNormalizedPerDecisionImportanceSampling,
    DiscreteSelfNormalizedDoublyRobust,
)
from ofrl.ope.cumulative_distribution_estimators_discrete import (
    DiscreteCumulativeDistributionDirectMethod,
    DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust,
)
from ofrl.ope.ope_continuous import ContinuousOffPolicyEvaluation
from ofrl.ope.estimators_continuous import (
    ContinuousDirectMethod,
    ContinuousTrajectoryWiseImportanceSampling,
    ContinuousPerDecisionImportanceSampling,
    ContinuousDoublyRobust,
    ContinuousSelfNormalizedTrajectoryWiseImportanceSampling,
    ContinuousSelfNormalizedPerDecisionImportanceSampling,
    ContinuousSelfNormalizedDoublyRobust,
)


__all__ = [
    "BaseOffPolicyEstimator",
    "BaseCumulativeDistributionOffPolicyEstimator",
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionOffPolicyEvaluation",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedPerDecisionImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
    "ContinuousOffPolicyEvaluation",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousPerDecisionImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousSelfNormalizedPerDecisionImportanceSampling",
    "ContinuousSelfNormalizedDoublyRobust",
]


__base__ = [
    "BaseOffPolicyEstimator",
    "BaseCumulativeDistributionOffPolicyEstimator",
]


__meta__ = [
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionOffPolicyEvaluation",
    "ContinuousOffPolicyEvaluation",
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


__cumulative__ = [
    "BaseCumulativeDistributionOffPolicyEstimator",
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
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
    "DiscreteCumulativeDistributionDirectMethod",
    "DiscreteCumulativeDistributionTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust",
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
]
