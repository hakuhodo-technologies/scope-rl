# import modules
from ofrl.policy.opl import OffPolicyLearning
# import models from d3rlpy
from d3rlpy.algos import CQL, IQL

from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.policy import DiscreteSoftmaxHead as SoftmaxHead

from typing import Dict, List, Any
from ofrl.utils import MinMaxActionScaler

import pickle
import torch
import numpy as np

def policy_learning_candidates(
    env: str,
    candidate_policy_params: Dict[str, List[float]],
    train_logged_dataset,
    random_state,
):
    if env.action_space == Box:
        # evaluation policies
        cql = CQL(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env.action_space.low,  # minimum value that policy can take
                maximum=env.action_space.high,  # maximum value that policy can take
            )
        )

        iql = IQL(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env.action_space.low,  # minimum value that policy can take
                maximum=env.action_space.high,  # maximum value that policy can take
            )
        )

        algorithms = [cql, iql]
        algorithms_name = ["cql", "iql"]

        # initialize OPL class
        opl = OffPolicyLearning(
            fitting_args={
                "n_steps": 10000,
                "scorers": {},
            }
        )

        # obtain base policies
        base_policies = opl.learn_base_policy(
            logged_dataset=train_logged_dataset,
            algorithms=algorithms,
            random_state=random_state,
        )

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "wb") as f:
            pickle.dump(base_policies, f)

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "rb") as f:
            base_policies = pickle.load(f)

        # policy wrapper
        policy_wrappers = {
            "gauss": (
                TruncatedGaussianHead, {
                    "sigma": np.array([candidate_policy_params[sigma]]),
                    "minimum": env.action_space.low,
                    "maximum": env.action_space.high,
                }
            )
        } 

        eval_policy = opl.apply_head(
            base_policies=base_policies,
            base_policies_name=algorithms_name,
            policy_wrappers=policy_wrappers,
            random_state=random_state,
        )   

    elif env.action_space==Discrete:

        # evaluation policies
        cql = CQL(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
        )

        algorithms = [cql]
        algorithms_name = ["cql"]

        # initialize OPL class
        opl = OffPolicyLearning(
            fitting_args={
                "n_steps": 10000,
                "scorers": {},
            }
        )

        # obtain base policies
        base_policies = opl.learn_base_policy(
            logged_dataset=train_logged_dataset,
            algorithms=algorithms,
            random_state=random_state,
        )

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "wb") as f:
            pickle.dump(base_policies, f)

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "rb") as f:
            base_policies = pickle.load(f)

        # policy wrapper
        policy_wrappers = {
            "eps": (
                EpsilonGreedyHead, {
                    "epsilon": candidate_policy_params[epsilon],
                    "n_actions": env.action_space.n,
                }
            ),
            
            "softmax": (
                SoftmaxHead, {
                    "tau": candidate_policy_params[tau],
                    "n_actions": env.action_space.n,
                }
            )
        }

        eval_policy = opl.apply_head(
            base_policies=base_policies,
            base_policies_name=algorithms_name,
            policy_wrappers=policy_wrappers,
            random_state=random_state,
        )   

    return eval_policy