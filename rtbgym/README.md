# RTBGym: A configurative reinforcement learning environment for real-time bidding research
<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [RTBGym: A reinforcement learning environment for real-time bidding research](#rtbgym-a-reinforcement-learning-environment-for-real-time-bidding-research)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Discrete Control](#discrete-control)
  - [Continuous Control](#continuous-control)
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

## Overview

*RTBGym* is an open-source simulation platform for Real-Time Bidding (RTB) of Display Advertising, which is written in Python. The simulator is particularly intended for reinforcement learning algorithms and follows [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface. We design RTBGym as a configurative environment so that researchers and practitioner can customize the environmental modules including `WinningPriceDistribution`, `ClickThroughRate`, and `ConversionRate`.

Note that, RTBGym is publicized under [SCOPE-RL](../) repository, which facilitates the implementation of offline reinforcement learning procedure.

### Basic Setting

In RTB, the objective of the RL agent is to maximize some KPIs (number of clicks or conversions) within an episode under given budget constraints. \
We often aim to achieve this goal by adjusting a parameter $\alpha$ to control the bid price as follows.

<p align="center">
$bid_{t,i} = \alpha \cdot r^{\ast}$,
</p>

where $r^{\ast}$ denotes a predicted or expected reward (KPIs).

We often formulate this RTB problem as the following Constrained Markov Decision Process (CMDP):
- `timestep`: One episode (a day or a week) consists of several timesteps (24 hours or seven days, for instance).
- `state`: We observe some feedback from the environment at each timestep, which includes the following.
  - timestep
  - remaining budget
  - impression level features (budget consumption rate, cost per mille of impressions, auction winning rate, reward) at the previous timestep
  - adjust rate (RL agent's decision making) at the previous timestep
- `action`: Agent chooses adjust rate parameter $\alpha$ to maximize KPIs.
- `reward`: Total number of clicks or conversions obtained during the timestep.
- `constraints`: The pre-determined episodic budget should not be exceeded.

The goal of RTB is to maximize the expected trajectory-wise reward under the budget constraint.

### Implementation

RTBGym provides two standardized RTB environment.
- `"RTBEnv-discrete-v0"`: Standard RTB environment with discrete action space.
- `"RTBEnv-continuous-v0"`: Standard RTB environment with continuous action space.

RTBGym consists of the following two environments.
- [RTBEnv](./envs/rtb.py#L24): The basic configurative environment with continuous action space.
- [CustomizedRTBEnv](./envs/wrapper_rtb.py#L15): The customized environment given action space and reward predictor.

RTBGym is configurative about the following three modules.
- [WinningPriceDistribution](./envs/simulator/function.py#L18): Class to define the winning price distribution of the auction bidding.
- [ClickThroughRate](./envs/simulator/function.py#L183): Class to define the click through rate of users.
- [ConversionRate](./envs/simulator/function.py#L393): Class to define the conversion rate of users.

Note that, users can customize the above modules by following the [abstract class](./envs/simulator/base.py). \
We also define the bidding function in the [Bidder](./envs/simulator/bidder.py#15) class and the auction simulation in the [Simulator](./envs/simulator/rtb_synthetic.py#23) class, respectively.

## Installation
RTBGym can be installed as a part of [SCOPE-RL](../) using Python's package manager `pip`.
```
pip install scope-rl
```

You can also install from source.
```bash
git clone https://github.com/hakuhodo-technologies/scope-rl
cd scope-rl
python setup.py install
```

## Usage

We provide an example usage of the standard and customized environment. \
The online/offlline RL and Off-Policy Evaluation examples are provides in [SCOPE-RL's README](../README.md).

### Standard RTBEnv

Our standard RTBEnv is available from `gym.make()`, following the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

```Python
# import rtbgym and gym
import rtbgym
import gym

# (1) standard environment for discrete action space
env = gym.make('RTBEnv-discrete-v0')

# (2) standard environment for continuous action space
env_ = gym.make('RTBEnv-continuous-v0')
```

The basic interaction is performed using only four lines of code as follows.

```Python
obs, info = env.reset(), False
while not done:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
```

Let's visualize the case with uniform random policy (in continuous action case). The discrete case also works in a similar manner.

```Python
# import from other libraries
from offlinegym.policy import OnlineHead
from d3rlpy.algos import RandomPolicy as ContinuousRandomPolicy
from d3rlpy.preprocessing import MinMaxActionScaler
import matplotlib.pyplot as plt

# define a random agent (for continuous action)
agent = OnlineHead(
    ContinuousRandomPolicy(
        action_scaler=MinMaxActionScaler(
            minimum=0.1,  # minimum value that policy can take
            maximum=10,  # maximum value that policy can take
        )
    ),
    name="random",
)
agent.build_with_env(env)

# (3) basic interaction for continuous action case
obs, info = env.reset()
done = False
# logs
remaining_budget = [obs[1]]
cumulative_reward = [0]

while not done:
    action = agent.predict_online(obs)
    obs, reward, done, truncated, info = env.step(action)
    # logs
    remaining_budget.append(obs[1])
    cumulative_reward.append(cumulative_reward[-1] + reward)

# visualize the result
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(remaining_budget[:-1], label='remaining budget')
ax2 = ax1.twinx()
ax2.plot(cumulative_reward[:-1], label='cumulative reward', color='tab:orange')
ax1.set_xlabel('timestep')
ax1.set_ylabel('remainig budget')
ax1.set_ylim(0, env.initial_budget + 100)
ax2.set_ylabel('reward (coversion)')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.show()
```
<div align="center"><img src="./images/basic_interaction.png" width="60%"/></div>
<figcaption>
<p align="center">
  Transition of the Remaining Budget and Cumulative Reward during a Single Episode
</p>
</figcaption>

Note that, while we use [SCOPE-RL](../README.md) and [d3rlpy](https://github.com/takuseno/d3rlpy) here, RTBGym is compatible with any other libraries working on the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

### Customized RTGEnv

Next, we describe how to customize the environment by instantiating the environment.

<details>
<summary>List of environmental configurations: (click to expand)</summary>

- `objective`: Objective KPIs of RTB, which is either "click" or "conversion".
- `cost_indicator`: Timing of arising costs, which is any of "impression", "click", and "conversion".
- `step_per_episode`: Number of timesteps in an episode.
- `initial_budget`: Initial budget (i.e., constraint) for an episode.
- `n_ads`: Number of ads used for auction bidding.
- `n_users`: Number of users used for auction bidding.
- `ad_feature_dim`: Dimension of the ad feature vectors.
- `user_feature_dim`: Dimension of the user feature vectors.
- `ad_feature_vector`: Feature vectors that characterizes each ad.
- `user_feature_vector`: Feature vectors that characterizes each user.
- `ad_sampling_rate`: Sampling probabilities to determine which ad (id) is used in each auction.
- `user_sampling_rate`: Sampling probabilities to determine which user (id) is used in each auction.
- `WinningPriceDistribution`: Winning price distribution of auctions.
- `ClickTroughRate`: Click through rate (i.e., click / impression).
- `ConversionRate`: Conversion rate (i.e., conversion / click).
- `standard_bid_price_distribution`: Distribution of the bid price whose average impression probability is expected to be 0.5.
- `minimum_standard_bid_price`: Minimum value for standard bid price.
- `search_volume_distribution`: Search volume distribution for each timestep.
- `minimum_search_volume`: Minimum search volume at each timestep.
- `random_state`: Random state.

</details>

```Python
from rtbgym import RTBEnv
env = RTBEnv(
    objective="click",  # maximize the number of total impressions
    cost_indicator="click",  # cost arises every time click occurs
    step_per_episode=14,  # 14 days as an episode
    initial_budget=5000,  # budget available for 14 dayas is 5000
    random_state=12345,
)
```

Specifically, users can define their own `WinningPriceDistribution`, `ClickThroughRate`, and `ConversionRate` as follows.

#### Example of Custom Winning Price Distribution
```Python
# import RTBGym modules
from rtbgym import BaseWinningPriceDistribution
from rtbgym.utils import NormalDistribution
# import other necessary stuffs
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import numpy as np

@dataclass
class CustomizedWinningPriceDistribution(BaseWinningPriceDistribution):
    """Initialization."""
    n_ads: int
    n_users: int
    ad_feature_dim: int
    user_feature_dim: int
    step_per_episode: int
    standard_bid_price_distribution: NormalDistribution = NormalDistribution(
        mean=50,
        std=5,
        random_state=12345,
    )
    minimum_standard_bid_price: Optional[Union[int, float]] = None
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

    def sample_outcome(
        self,
        bid_prices: np.ndarray,
        **kwargs,
    ) -> Tuple[np.ndarray]:
        """Stochastically determine impression and second price for each auction."""
        # sample winning price from simple normal distribution
        winning_prices = self.random_.normal(
            loc=self.standard_bid_price,
            scale=self.standard_bid_price / 5,
            size=bid_prices.shape,
        )
        impressions = winning_prices < bid_prices
        return impressions.astype(int), winning_prices.astype(int)

    @property
    def standard_bid_price(self):
        return self.standard_bid_price_distribution.mean
```

#### Example of Custom ClickThroughRate (and Conversion Rate)
```Python
from rtbgym import BaseClickAndConversionRate
from rtbgym.utils import sigmoid

@dataclass
class CustomizedClickThroughRate(BaseClickAndConversionRate):
    """Initialization."""
    n_ads: int
    n_users: int
    ad_feature_dim: int
    user_feature_dim: int
    step_per_episode: int
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.ad_coef = self.random_.normal(
            loc=0.0,
            scale=0.5,
            size=(self.ad_feature_dim, 10),
        )
        self.user_coef = self.random_.normal(
            loc=0.0,
            scale=0.5,
            size=(self.user_feature_dim, 10),
        )

    def calc_prob(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Calculate CTR (i.e., click per impression)."""
        ad_latent = ad_feature_vector @ self.ad_coef
        user_latent = user_feature_vector @ self.user_coef
        ctrs = sigmoid((ad_latent * user_latent).mean(axis=1))
        return ctrs

    def sample_outcome(
        self,
        ad_ids: np.ndarray,
        user_ids: np.ndarray,
        ad_feature_vector: np.ndarray,
        user_feature_vector: np.ndarray,
        timestep: Union[int, np.ndarray],
    ) -> np.ndarray:
        """Stochastically determine whether click occurs in impression=True case."""
        ctrs = self.calc_prob(
            timestep=timestep,
            ad_ids=ad_ids,
            user_ids=user_ids,
            ad_feature_vector=ad_feature_vector,
            user_feature_vector=user_feature_vector,
        )
        clicks = self.random_.rand(len(ad_ids)) < ctrs
        return clicks.astype(int)
```
Note that, custom conversion rate can be defined in a simmilar manner.

### Wrapper class for custom bidding setup
To customize the bidding setup, we also provide `CustomizedRTBEnv`.

`CustomizedRTBEnv` enables discretization or re-definition of the action space.
In addition, users can set their own `reward_predictor`.

<details>
<summary>List of arguments: (click to expand)</summary>

- `original_env`: Original RTB Environment.
- `reward_predictor`: A machine learning model to predict the reward to determine the bidding price.
- `scaler`: Scaling factor (constant value) used for bid price determination. (None for the auto-fitting)
- `action_min`: Minimum value of adjust rate.
- `action_max`: Maximum value of adjust rate.
- `action_type`: Action type of the RL agent, which is either "discrete" or "continuous".
- `n_actions`: Number of "discrete" actions.
- `action_meaning`: Mapping function of agent action index to the actual "discrete" action to take.

</details>

```Python
from rtbgym import CustomizedRTBEnv
custom_env = CustomizedRTBEnv(
    original_env=env,
    reward_predictor=None,  # use ground-truth (expected) reward as a reward predictor (oracle)
    action_type="discrete",
)
```

More examples are available at [quickstart/rtb_synthetic_customize_env.ipynb](./examples/quickstart/rtb_synthetic_customize_env.ipynb). \
The statistics of the environment is also visualized at [quickstart/rtb_synthetic_data_collection.ipynb](./examples/quickstart/rtb_synthetic_data_collection.ipynb).

Finally, example usages for online/offline RL and OPE/OPS studies are available at [quickstart/rtb_synthetic_discrete_basic.ipynb](./examples/quickstart/rtb_synthetic_discrete_basic.ipynb) (discrete action space) and [quickstart/rtb_synthetic_continuous_basic.ipynb](./examples/quickstart/rtb_synthetic_continuous_basic.ipynb) (continuous action space).

## Citation

If you use our software in your work, please cite our paper:

Haruka Kiyohara, Kosuke Kawakami, Yuta Saito.<br>
**Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation**<br>
(RecSys'21 SimuRec workshop)<br>
[https://arxiv.org/abs/2109.08331](https://arxiv.org/abs/2109.08331)

Bibtex:
```
@article{kiyohara2021accelerating,
  title={Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation},
  author={Kiyohara, Haruka and Kawakami, Kosuke and Saito, Yuta},
  journal={arXiv preprint arXiv:2109.08331},
  year={2021}
}
```

## Contribution
Any contributions to RTBGym are more than welcome!
Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines how to contribute the project.

## License

This project is licensed under Apache 2.0 license - see [LICENSE](../LICENSE) file for details.

## Project Team

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**)
- Ren Kishimoto (Tokyo Institute of Technology)
- Kosuke Kawakami (HAKUHODO Technologies Inc.)
- Ken Kobayashi (Tokyo Institute of Technology)
- Kazuhide Nakata (Tokyo Institute of Technology)
- [Yuta Saito](https://usait0.com/en/) (Cornell University)


## Contact

For any question about the paper and software, feel free to contact: hk844@cornell.edu

## References

<details>
<summary><strong>Papers </strong>(click to expand)</summary>

1. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

2. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

3. Di Wu, Xiujun Chen, Xun Yang, Hao Wang, Qing Tan, Xiaoxun Zhang, Jian Xu, and Kun Gai. [Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising](https://arxiv.org/abs/1802.08365). In *Proceedings of the 27th ACM International Conference on Information and Knowledge Management*, 1443-1451, 2018.

4. Jun Zhao, Guang Qiu, Ziyu Guan, Wei Zhao, and Xiaofei He. [Deep Reinforcement Learning for Sponsored Search Real-time Bidding](https://arxiv.org/abs/1803.00259). In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1021-1030, 2018.

5. Wen-Yuan Zhu, Wen-Yueh Shih, Ying-Hsuan Lee, Wen-Chih Peng, and Jiun-Long Huang. [A Gamma-based Regression for Winning Price Estimation in Real-Time Bidding Advertising](https://ieeexplore.ieee.org/document/8258095). In *IEEE International Conference on Big Data*, 1610-1619, 2017.

</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is inspired by the following three packages.
- **AuctionGym** -- an RL environment for online advertising auctions: [[github](https://github.com/amzn/auction-gym)] [[paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym)]
- **RecoGym**  -- an RL environment for recommender systems: [[github](https://github.com/criteo-research/reco-gym)] [[paper](https://arxiv.org/abs/1808.00720)]
- **RecSim** -- a configurative RL environment for recommender systems: [[github](https://github.com/google-research/recsim)] [[paper](https://arxiv.org/abs/1909.04847)]
- **FinRL** -- an RL environment for finance: [[github](https://github.com/AI4Finance-Foundation/FinRL)] [[paper](https://arxiv.org/abs/2011.09607)]

</details>

