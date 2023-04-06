
import pickle
import torch
from typing import Dict, List, Any
import gym
from gym.spaces import Box, Discrete
from pathlib import Path
from typing import Optional

from ofrl.utils import MinMaxScaler, MinMaxActionScaler
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory

# import ope modules from OFRL
from ofrl.ope import CreateOPEInput
from ofrl.ope import OffPolicyEvaluation as OPE
from ofrl.ope import ContinuousDirectMethod as ContinuousDM
from ofrl.ope import ContinuousPerDecisionImportanceSampling as ContinuousPDIS
from ofrl.ope import ContinuousDoublyRobust as ContinuousDR
from ofrl.ope import ContinuousStateActionMarginalImportanceSampling as ContinuousMIS

from ofrl.ope import DiscreteDirectMethod as DiscreteDM
from ofrl.ope import DiscretePerDecisionImportanceSampling as DiscretePDIS
from ofrl.ope import DiscreteDoublyRobust as DiscreteDR
from ofrl.ope import DiscreteStateActionMarginalImportanceSampling as DiscreteMIS
from ofrl.ope import OffPolicySelection

def ops_topk_evaluation(
    env: gym.Env,
    test_logged_dataset: Dict[str, Any],
    behavior_policy,
    eval_policy,
    random_state: Optional[int] = None,
):
    # first, prepare OPE inputs
    if isinstance(env.action_space, Box):
        prep = CreateOPEInput(
            env=env,
            model_args={
                "fqe": {
                    "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                    "q_func_factory": MeanQFunctionFactory(),
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                }
            },
            state_scaler=MinMaxScaler(
                minimum=test_logged_dataset.get(behavior_policy_name=behavior_policy.name, dataset_id=0)["state"].min(axis=0),
                maximum=test_logged_dataset.get(behavior_policy_name=behavior_policy.name, dataset_id=0)["state"].max(axis=0),
            ),
            # if env.action_space == Box:
                action_scaler=MinMaxActionScaler(
                    minimum=env.action_space.low,  # minimum value that policy can take
                    maximum=env.action_space.high,  # maximum value that policy can take
                ),
                sigma=0.1,
        )

    elif isinstance(env.action_space, Discrete):
        prep = CreateOPEInput(
            env=env,
            model_args={
                "fqe": {
                    "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
                    "q_func_factory": MeanQFunctionFactory(),
                    "learning_rate": 1e-4,
                    "use_gpu": torch.cuda.is_available(),
                }
            },
            state_scaler=MinMaxScaler(
                minimum=test_logged_dataset.get(behavior_policy_name=behavior_policy.name, dataset_id=0)["state"].min(axis=0),
                maximum=test_logged_dataset.get(behavior_policy_name=behavior_policy.name, dataset_id=0)["state"].max(axis=0),
            ),
        )

    # takes time
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=test_logged_dataset,
        evaluation_policies=eval_policy,
        require_value_prediction=True,
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )   

    with open("logs/ope_input_dict_continuous_multiple_datasets.pkl", "wb") as f:
        pickle.dump(input_dict, f)   
    
    with open("logs/ope_input_dict_continuous_multiple_datasets.pkl", "rb") as f:
        input_dict = pickle.load(f)
        
    if isinstance(env.action_space, Box):
        ope = OPE(
            logged_dataset=test_logged_dataset,
            ope_estimators=[ContinuousDM(), ContinuousPDIS(), ContinuousDR(), ContinuousMIS()],
        )

    elif isinstance(env.action_space, Discrete):
        ope = OPE(
            logged_dataset=test_logged_dataset,
            ope_estimators=[DiscreteDM(), DiscretePDIS(), DiscreteDR(), DiscreteMIS()],
        )


    ops = OffPolicySelection(
        ope=ope,
    )

    true_selection_result = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_variance=True,
        return_lower_quartile=True,
        return_conditional_value_at_risk=True,
        return_by_dataframe=True,
    )

    topk_metric_df = ops.obtain_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        compared_estimators=["dm", "pdis", "dr"],
        # compared_estimators=["dm", "pdis", "dr", "mis"],
        return_safety_violation_rate=True,
        safety_threshold=2.0,
        relative_safety_criteria=1.0,
        return_by_dataframe=True,
    )

    with open("logs/topk_metric_df.pkl", "wb") as f:
        pickle.dump(topk_metric_df, f)
    
    fig_path = Path("./images")

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        compared_estimators=["dm", "pdis", "dr"],
        # compared_estimators=["dm", "pdis", "dr", "mis"],
        visualize_ci=True,
        metrics=["best", "worst", "mean", "safety_violation_rate"],
        safety_threshold=2.0,
        relative_safety_criteria=1.0,
        legend=True,
        fig_dir=fig_path,
        fig_name="topk_policy_value_standard_ope.png",
        random_state=random_state,
    )
