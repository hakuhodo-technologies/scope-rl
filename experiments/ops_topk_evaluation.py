

# import ope modules from OFRL
from ofrl.ope import OffPolicyEvaluation as OPE
from ofrl.ope import ContinuousDirectMethod as DM
from ofrl.ope import ContinuousTrajectoryWiseImportanceSampling as TIS
from ofrl.ope import ContinuousPerDecisionImportanceSampling as PDIS
from ofrl.ope import ContinuousDoublyRobust as DR
from ofrl.ope import ContinuousStateActionMarginalImportanceSampling as MIS

from ofrl.ope import CumulativeDistributionOffPolicyEvaluation as CumulativeDistributionOPE
from ofrl.ope import ContinuousCumulativeDistributionDirectMethod as CD_DM
from ofrl.ope import ContinuousCumulativeDistributionTrajectoryWiseImportanceSampling as CD_IS
from ofrl.ope import ContinuousCumulativeDistributionTrajectoryWiseDoublyRobust as CD_DR
from ofrl.ope import ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseImportanceSampling as CD_SNIS
from ofrl.ope import ContinuousCumulativeDistributionSelfNormalizedTrajectoryWiseDoublyRobust as CD_SNDR

from ofrl.ope import OffPolicySelection

def ops_topk_evaluation(
        
):
    # first, prepare OPE inputs
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
            minimum=test_logged_dataset.get(behavior_policy_name=behavior_policies[0].name, dataset_id=0)["state"].min(axis=0),
            maximum=test_logged_dataset.get(behavior_policy_name=behavior_policies[0].name, dataset_id=0)["state"].max(axis=0),
        ),
        action_scaler=MinMaxActionScaler(
            minimum=env.action_space.low,  # minimum value that policy can take
            maximum=env.action_space.high,  # maximum value that policy can take
        ),
        sigma=0.1,
    )

    # takes time
    input_dict = prep.obtain_whole_inputs(
        logged_dataset=test_logged_dataset,
        evaluation_policies=eval_policies,
        require_value_prediction=True,
        n_trajectories_on_policy_evaluation=100,
        random_state=random_state,
    )   

    with open("logs/ope_input_dict_continuous_multiple_datasets.pkl", "wb") as f:
        pickle.dump(input_dict, f)   
      
    with open("logs/ope_input_dict_continuous_multiple_datasets.pkl", "rb") as f:
        input_dict = pickle.load(f)

    ope = OPE(
        logged_dataset=test_logged_dataset,
        ope_estimators=[DM(), PDIS(), DR(), MIS()],
    )
  
    cd_ope = CumulativeDistributionOPE(
        logged_dataset=test_logged_dataset,
        ope_estimators=[
            CD_DM(estimator_name="cdf_dm"), 
            CD_IS(estimator_name="cdf_is"), 
            CD_DR(estimator_name="cdf_dr"), 
            CD_SNIS(estimator_name="cdf_snis"), 
            CD_SNDR(estimator_name="cdf_sndr"),
        ],
    )

    ops = OffPolicySelection(
      ope=ope,
      cumulative_distribution_ope=cd_ope,
    )

    true_selection_result = ops.obtain_true_selection_result(
        input_dict=input_dict,
        return_variance=True,
        return_lower_quartile=True,
        return_conditional_value_at_risk=True,
        safety_threshold=1.0,
        return_by_dataframe=True,
    )

    ops.visualize_topk_policy_value_selected_by_standard_ope(
        input_dict=input_dict,
        compared_estimators=["dm", "pdis", "dr", "mis"],
        visualize_ci=True,
        metrics=["best", "worst", "mean", "safety_violation_rate"],
        safety_threshold=2.0,
        legend=True,
        fig_dir=,
        fig_name="topk_policy_value_standard_ope.png",
        random_state=random_state,
    )
