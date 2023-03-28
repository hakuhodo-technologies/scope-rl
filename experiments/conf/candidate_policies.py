# write here
# reference: https://github.com/sony/pyIEOE/blob/master/benchmark/multiclass/conf/base_eval_policy_params.py
# also refer to conf/behavior_policy.py

from d3rlpy.algos import SAC
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory


candidate_policy_model_confs = {
    "cql": {
        "actor_encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
        "critic_encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
        "q_func_factory": MeanQFunctionFactory(),
    },
    "iql": {
        "actor_encoder_factory": VectorEncoderFactory(hidden_units=[50, 10]),"critic_encoder_factory" : VectorEncoderFactory(hidden_units=[50, 10]),
    },
}
