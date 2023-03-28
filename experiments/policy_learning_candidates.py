# import modules
from ofrl.policy.opl import OffPolicyLearning
# import models from d3rlpy
from d3rlpy.algos import CQL, IQL

def policy_learning_candidates(
    env: str,
    candidate_policy_params: Dict[str, List[float]],
    candidate_epsilons: List[float], 
    train_logged_dataset,
    n_random_state,
):
    # evaluation policies
    cql_b1 = CQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        q_func_factory=MeanQFunctionFactory(),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )
    cql_b2 = CQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
        q_func_factory=MeanQFunctionFactory(),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )
    cql_b3 = CQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
        q_func_factory=MeanQFunctionFactory(),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )
    iql_b1 = IQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )
    iql_b2 = IQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[100]),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )
    iql_b3 = IQL(
        actor_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
        critic_encoder_factory=VectorEncoderFactory(hidden_units=[50, 10]),
        use_gpu=torch.cuda.is_available(),
        action_scaler=MinMaxActionScaler(
            minimum=env_.action_space.low,  # minimum value that policy can take
            maximum=env_.action_space.high,  # maximum value that policy can take
        )
    )

    algorithms = [cql_b1, cql_b2, cql_b3, iql_b1, iql_b2, iql_b3]
    algorithms_name = ["cql_b1", "cql_b2", "cql_b3", "iql_b1", "iql_b2", "iql_b3"]

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
        "gauss_05": (
            TruncatedGaussianHead, {
                "sigma": np.array([0.5]),
                "minimum": env.action_space.low,
                "maximum": env.action_space.high,
            }
        ),
        "gauss_10": (
            TruncatedGaussianHead, {
                "sigma": np.array([1.0]),
                "minimum": env.action_space.low,
                "maximum": env.action_space.high,
            }
        ),
    } 

    eval_policies = opl.apply_head(
        base_policies=base_policies,
        base_policies_name=algorithms_name,
        policy_wrappers=policy_wrappers,
        random_state=random_state,
    )   

    return eval_policies