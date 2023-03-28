# write here
# reference: https://github.com/sony/pyIEOE/blob/master/benchmark/multiclass/conf/base_eval_policy_params.py

from d3rlpy.algos import SAC
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory

behavior_policy_model_confs = {
    "sac": {
        "actor_encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
        "critic_encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),
        "q_func_factory": MeanQFunctionFactory(),
    },
    "ddqn": {
        "encoder_factory": VectorEncoderFactory(hidden_units=[30, 30]),"q_func_factory" : MeanQFunctionFactory(),
        "target_update_interval" : 100,
    },
}

replay_buffer_confs = {
    "max_len": 10000,
}