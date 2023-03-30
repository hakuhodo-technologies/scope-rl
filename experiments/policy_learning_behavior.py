from typing import Optional

# import OFRL modules
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.utils import MinMaxActionScaler

# import d3rlpy algorithms
from d3rlpy.algos import RandomPolicy
# from d3rlpy.preprocessing import MinMaxActionScaler
from ofrl.utils import MinMaxActionScaler
from ofrl.utils import OldGymAPIWrapper

# import from other libraries
import gym
from gym.spaces import Box, Discrete
import torch
import numpy as np
from typing import Dict

# behavior policy
from d3rlpy.algos import SAC
from d3rlpy.algos import DoubleDQN
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.online.buffers import ReplayBuffer

import gym

def policy_learning_behavior(
    env: gym.Env,
    behavior_policy_params: Dict[str, float],
    random_state: Optional[int] = None,
) :
    env_ = OldGymAPIWrapper(env)

    if isinstance(env_.action_space, Box):
        # model
        sac = SAC(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env_.action_space.low,  
                maximum=env_.action_space.high,  
            ),
        )
        # setup replay buffer
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )

        # start training
        # skip if there is a pre-trained model
        sac.fit_online(
            env_,
            buffer,
            eval_env=env_,
            # n_steps=10000,
            # n_steps_per_epoch=100,
            # update_start_step=100,
            n_steps=1000,
            n_steps_per_epoch=100,
            update_start_step=100,
        )

        # save model
        sac.save_model("d3rlpy_logs/sac.pt")

        # reload model
        sac.build_with_env(env_)
        sac.load_model("d3rlpy_logs/sac.pt")

        behavior_policy = TruncatedGaussianHead(
            sac, 
            minimum=env_.action_space.low,
            maximum=env_.action_space.high,
            sigma=np.array([behavior_policy_params["sigma"]]),
            name="sac",
            random_state=random_state,
        )
    
    elif isinstance(env_.action_space, Discrete):
        # model
        ddqn = DoubleDQN(
            encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            target_update_interval=100,
            use_gpu=torch.cuda.is_available(),
        )
        # replay buffer
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env_,
        )
        # explorers
        explorer = LinearDecayEpsilonGreedy(
            start_epsilon=1.0,
            end_epsilon=0.1,
            duration=1000,
        )

        # start training
        # skip if there is a pre-trained model
        ddqn.fit_online(
            env_,
            buffer,
            explorer=explorer,
            eval_env=env_,
            n_steps=100000,
            n_steps_per_epoch=100,
            update_start_step=100,
            # n_steps=100000,
            # n_steps_per_epoch=1000,
            # update_start_step=1000,
        )

        # save model
        ddqn.save_model("d3rlpy_logs/ddqn.pt")

        # reload model
        ddqn.build_with_env(env_)
        ddqn.load_model("d3rlpy_logs/ddqn.pt")

        behavior_policy = EpsilonGreedyHead(
            ddqn, 
            n_actions=env_.action_space.n,
            epsilon=behavior_policy_params["epsilon"],
            name="ddqn",
            random_state=random_state,
        )
        
    return behavior_policy