from ofrl.ope.ops import OffPolicySelection
from ofrl.ope.input import CreateOPEInput
from ofrl.ope.estimators_base import (
    BaseOffPolicyEstimator,
    BaseCumulativeDistributionalOffPolicyEstimator,
)
from ofrl.ope.ope_discrete import (
    DiscreteOffPolicyEvaluation,
    DiscreteCumulativeDistributionalOffPolicyEvaluation,
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
from ofrl.ope.cumulative_distributional_estimators_discrete import (
    DiscreteCumulativeDistributionalDirectMethod,
    DiscreteCumulativeDistributionalTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionalTrajectoryWiseDoublyRobust,
    DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseImportanceSampling,
    DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseDoublyRobust,
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
    "BaseCumulativeDistributionalOffPolicyEstimator",
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionalOffPolicyEvaluation",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscretePerDecisionImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedPerDecisionImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "DiscreteCumulativeDistributionalDirectMethod",
    "DiscreteCumulativeDistributionalTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseDoublyRobust",
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
    "BaseCumulativeDistributionalOffPolicyEstimator",
]


__meta__ = [
    "OffPolicySelection",
    "CreateOPEInput",
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionalOffPolicyEvaluation",
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
    "BaseCumulativeDistributionalOffPolicyEstimator",
    "DiscreteCumulativeDistributionalDirectMethod",
    "DiscreteCumulativeDistributionalTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseDoublyRobust",
]


__discrete__ = [
    "DiscreteOffPolicyEvaluation",
    "DiscreteCumulativeDistributionalOffPolicyEvaluation",
    "DiscreteDistributionallyRobustOffPolicyEvaluation",
    "DiscreteDirectMethod",
    "DiscreteTrajectoryWiseImportanceSampling",
    "DiscreteStepWiseImportanceSampling",
    "DiscreteDoublyRobust",
    "DiscreteSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteSelfNormalizedStepWiseImportanceSampling",
    "DiscreteSelfNormalizedDoublyRobust",
    "DiscreteCumulativeDistributionalDirectMethod",
    "DiscreteCumulativeDistributionalTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalTrajectoryWiseDoublyRobust",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseImportanceSampling",
    "DiscreteCumulativeDistributionalSelfNormalizedTrajectoryWiseDoublyRobust",
]


__continuous__ = [
    "ContinuousOffPolicyEvaluation",
    "ContinuousCumulativeDistributionalOffPolicyEvaluation",
    "ContinuousDistributionallyRobustOffPolicyEvaluation",
    "ContinuousDirectMethod",
    "ContinuousTrajectoryWiseImportanceSampling",
    "ContinuousStepWiseImportanceSampling",
    "ContinuousDoublyRobust",
    "ContinuousSelfNormalizedTrajectoryWiseImportanceSampling",
    "ContinuousSelfNormalizedStepWiseImportanceSampling",
    "ContinuousSelfNormalizedDoublyRobust",
]
