"""Off-Policy Evaluation Class to Streamline OPE."""
from rtb_reinforcement_learing._gym.utils import check_logged_dataset
from rtb_reinforcement_learing._gym.ope.estimators import BaseOffPolicyEstimator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns

from d3rlpy.ope import DiscreteFQE
from d3rlpy.ope import FQE as ContinuousFQE
from d3rlpy.algos import AlgoBase

from _gym.ope import BaseOffPolicyEstimator
from _gym.types import LoggedDataset
from _gym.utils import (
    convert_logged_dataset_into_MDPDataset,
    check_base_model_args,
)


@dataclass
class OffPolicyEvaluation:
    """Class to conduct OPE by multiple estimators simultaneously.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    ope_estimators: List[BaseOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `_gym.ope.BaseOffPolicyEstimator`.

    Examples
    ----------

    """

    logged_dataset: LoggedDataset
    ope_estimators: List[BaseOffPolicyEstimator]

    def __post__init__(self) -> None:
        "Initialize class."
        self.ope_estimators_ = dict()
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator


@dataclass
class PrepareOPEInput:
    """Class to prepare OPE inputs.

    Parameters
    -----------
    logged_dataset: LoggedDataset
        Logged dataset used to conduct OPE.

    base_model_args: Optional[Dict[str, Any]], default = None
        Arguments of baseline Fitted Q Evaluation (FQE) model.

    """

    logged_dataset: LoggedDataset
    base_model_args: Optional[Dict[str, Any]] = None
    use_base_model: bool = False

    def __post__init__(self) -> None:
        "Initialize class."
        self.action_type = self.logged_dataset["action_type"]
        self.mdp_dataset = convert_logged_dataset_into_MDPDataset(self.logged_dataset)
        if self.use_base_model:
            if self.base_model_args is None:
                self.base_model_args = {
                    "n_epochs": 200,
                    "q_func_factory": "qr",
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                    "encoder_params": {"hidden_units": [20]},
                }
            check_base_model_args(
                dataset=self.mdp_dataset,
                args=self.base_model_args,
                action_type=self.action_type,
            )

    def construct_FQE(
        self,
        evaluation_policy: AlgoBase,
        validation_size: float = 0.2,
        n_epochs: Optional[int] = None,
        n_steps: Optional[int] = None,  # should be more than n_steps_per_epoch
        n_steps_per_epoch: int = 10000,
    ):
        if n_epochs is None and n_steps is None:
            n_steps = n_steps_per_epoch

        self.base_model_args["algo"] = evaluation_policy

        if self.action_type == "discrete":
            self.fqe = DiscreteFQE(**self.base_model_args)
        else:
            self.fqe = ContinuousFQE(**self.base_model_args)
        self.fqe.fit()
