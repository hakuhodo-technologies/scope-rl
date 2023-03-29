# import OFRL modules
import ofrl
from basicgym import BasicEnv
from ofrl.dataset import SyntheticDataset
from ofrl.policy import OnlineHead, ContinuousEvalHead
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.ope.online import (
    calc_on_policy_policy_value,
    visualize_on_policy_policy_value,
)
from ofrl.utils import MinMaxScaler, MinMaxActionScaler

# import d3rlpy algorithms
from d3rlpy.algos import RandomPolicy
# from d3rlpy.preprocessing import MinMaxScaler, MinMaxActionScaler
from ofrl.utils import MinMaxScaler, MinMaxActionScaler

# import from other libraries
import gym
import torch
from sklearn.model_selection import train_test_split

import pickle
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

from typing import Dict, List, Any

# behavior policy
from d3rlpy.algos import SAC
from d3rlpy.algos import DoubleDQN
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.online.buffers import ReplayBuffer



def policy_learning_behavior(
    env: str,
    behavior_policy_params: Dict[str, float],
    behavior_policy_model_confs: Dict[str, Any],
    random_state,
):
    
    if env.action_space == Discrete:

        # model
        sac = SAC(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env.action_space.low,  
                maximum=env.action_space.high,  
            ),
        )
        # setup replay buffer
        buffer = ReplayBuffer(
            maxlen=10000,
            env=env,
        )

        # start training
        # skip if there is a pre-trained model
        sac.fit_online(
            env,
            buffer,
            eval_env=env,
            n_steps=10000,
            n_steps_per_epoch=100,
            update_start_step=100,
        )

        # save model
        sac.save_model("d3rlpy_logs/sac.pt")

        # reload model
        sac.build_with_env(env)
        sac.load_model("d3rlpy_logs/sac.pt")

        behavior_policy = TruncatedGaussianHead(
            sac, 
            minimum=env.action_space.low,
            maximum=env.action_space.high,
            sigma=np.array([behavior_policy_params[sigma]]),
            name="sac",
            random_state=random_state,
        )
    
    elif env.action_space == Box:
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
            env=env,
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
            env,
            buffer,
            explorer=explorer,
            eval_env=env,
            n_steps=100000,
            n_steps_per_epoch=1000,
            update_start_step=1000,
        )

        # save model
        ddqn.save_model("d3rlpy_logs/ddqn.pt")

        # reload model
        ddqn.build_with_env(env)
        ddqn.load_model("d3rlpy_logs/ddqn.pt")

        behavior_policy = EpsilonGreedyHead(
            ddqn, 
            n_actions=env.action_space.n,
            epsilon=behavior_policy_params[epsilon],
            name="ddqn",
            random_state=random_state,
        )
        
    return behavior_policy