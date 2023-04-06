# import modules
from ofrl.policy.opl import OffPolicyLearning
from ofrl.utils import OldGymAPIWrapper
from ofrl.policy import OnlineHead, ContinuousEvalHead

# import models from d3rlpy
from d3rlpy.algos import CQL, IQL
from d3rlpy.algos import DiscreteCQL
from d3rlpy.dataset import MDPDataset

from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from ofrl.policy import ContinuousTruncatedGaussianHead as TruncatedGaussianHead
from ofrl.policy import DiscreteEpsilonGreedyHead as EpsilonGreedyHead
from ofrl.policy import DiscreteSoftmaxHead as SoftmaxHead

from typing import Dict, List, Any
from gym.spaces import Box, Discrete
from ofrl.utils import MinMaxActionScaler

import pickle
import torch
import numpy as np
import gym
from typing import Optional
from sklearn.model_selection import train_test_split


def select_evaluation_policy(
    env: gym.Env,
    train_logged_dataset: Dict[str, Any],
    random_state: Optional[int] = None,
):
    env_ = OldGymAPIWrapper(env)

    # transform offline dataset for d3rlpy
    offlinerl_dataset = MDPDataset(
        observations=train_logged_dataset["state"],
        actions=train_logged_dataset["action"],
        rewards=train_logged_dataset["reward"],
        terminals=train_logged_dataset["done"],
        episode_terminals=train_logged_dataset["done"],
        discrete_action=False,
    )
    train_episodes, test_episodes = train_test_split(offlinerl_dataset, test_size=0.2, 
    random_state=random_state)


    if isinstance(env_.action_space, Box):
        # evaluation policies
        cql = CQL(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env_.action_space.low,  # minimum value that policy can take
                maximum=env_.action_space.high,  # maximum value that policy can take
            ),
        )

        iql = IQL(
            actor_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            critic_encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            use_gpu=torch.cuda.is_available(),
            action_scaler=MinMaxActionScaler(
                minimum=env_.action_space.low,  # minimum value that policy can take
                maximum=env_.action_space.high,  # maximum value that policy can take
            ),
        )

        cql.fit(
            train_episodes,
            eval_episodes=test_episodes,
            n_steps=10000,
            # n_steps=10000,
            scorers={},
        )

        # define evaluation policies (should be deterministic policy)
        cql_ = ContinuousEvalHead(
            base_policy=cql,
            name="cql",
        )
        evaluation_policy = [cql_]

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "wb") as f:
            pickle.dump(base_policies, f)

        with open("d3rlpy_logs/multiple_continuous_base_policies.pkl", "rb") as f:
            base_policies = pickle.load(f)


    elif isinstance(env_.action_space, Discrete):

        # evaluation policies
        cql_b1 = DiscreteCQL(
            encoder_factory=VectorEncoderFactory(hidden_units=[30, 30]),
            q_func_factory=MeanQFunctionFactory(),
            use_gpu=torch.cuda.is_available(),
        )


    return evaluation_policy
