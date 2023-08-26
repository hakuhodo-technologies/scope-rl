# RECGym: A configurative reinforcement learning environment for recommender system
<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [RECGym: A reinforcement learning environment for recommender system](#RECGym-a-reinforcement-learning-environment-for-recommender-system)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

## Overview

*RECGym* is an open-source Python platform for RL simulations on a recommender system (REC) environment. The simulator is particularly intended for reinforcement learning algorithms and follows [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface. We design RECGym as a configurative environment so that researchers and practitioners can customize the environmental modules including `UserModel`((i.e. `user_preference_dynamics` and `reward_function`) based on their own research purposes.

Note that RECGym is publicized under [SCOPE-RL](../) repository, which facilitates the implementation of the offline reinforcement learning procedure.

### Basic Setting

In a recommender system application, the objective of an RL agent is to maximize the expected cumulative reward such as user engagement. \
We often formulate this recommendation problem as the following (Partially Observable) Markov Decision Process ((PO)MDP):
- `state`:
   - A vector representing user preference. The preference changes over time in an episode depending on the actions presented by the RL agent.
   - When the true state is unobservable, the agent uses observations instead of the state.
- `action`: Indicating which item to present to the user.
- `reward`: User engagement signal as a reward. Either binary or continuous.

### Implementation

RECGym provides a recommender environment.
- `"RECEnv-v0"`: Standard recommender environment.

RECGym consists of the following environment.
- [RECEnv](./envs/rec.py#L14): The basic configurative environment.

RECGym is configurative about the following module.
- [UserModel](./envs/simulator/function.py#L13): Class to define the user model of the recommender system.

Note that users can customize the above modules by following the [abstract class](./envs/simulator/base.py).

## Installation
RECGym can be installed as a part of [SCOPE-RL](../) using Python's package manager `pip`.
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
The online/offline RL and Off-Policy Evaluation examples are provided in [SCOPE-RL's README](../README.md).

### Standard RECEnv

Our standard RECEnv is available from `gym.make()`, following the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

```Python
# import RECGym and gym
import recgym
import gym

# (1) standard environment
env = gym.make('RECEnv-v0')
```

The basic interaction is performed using only four lines of code as follows.

```Python
obs, info = env.reset()
while not done:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
```

Let's visualize the case with the uniform random policy.

```Python
# import from other libraries
from scope_rl.policy import OnlineHead
from d3rlpy.algos import DiscreteRandomPolicy

# define a random agent
agent = OnlineHead(
    DiscreteRandomPolicy(),
    name="random",
)
agent.build_with_env(env)

# (2) basic interaction
obs, info = env.reset()
done = False
# logs
reward_list = []

while not done:
    action = agent.sample_action_online(obs)
    obs, reward, done, truncated, info = env.step(action)
    # logs
    reward_list.append(reward)


# visualize the result
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(reward_list[:-1], label='reward', color='tab:orange')
ax1.set_xlabel('timestep')
ax1.set_ylabel('reward')
ax1.legend(loc='upper left')
plt.show()
```
<div align="center"><img src="./images/basic_interaction.png" width="60%"/></div>
<figcaption>
<p align="center">
  The Transition of the Reward during a Single Episode
</p>
</figcaption>

Note that while we use [SCOPE-RL](../README.md) and [d3rlpy](https://github.com/takuseno/d3rlpy) here, RECGym is compatible with any other libraries working on the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

### Customized RECEnv

Next, we describe how to customize the environment by instantiating the environment.

<details>
<summary>List of environmental configurations: (click to expand)</summary>

- `step_per_episode`: Number of timesteps in an episode.
- `n_items`: Number of items used in the recommender system.
- `n_users`: Number of users used in the recommender system.
- `item_feature_dim`: Dimension of the item feature vectors.
- `user_feature_dim`: Dimension of the user feature vectors.
- `item_feature_vector`: Feature vectors that characterize each item.
- `user_feature_vector`: Feature vectors that characterize each user.
- `reward_type`: Reward type.
- `reward_std`: Noise level of the reward. Applicable only when reward_type is "continuous".
- `obs_std`: Noise level of the state observation.
- `UserModel`: User model that defines the user prefecture dynamics and reward function.
- `random_state`: Random state.

</details>

```Python
from recgym import RECEnv
env = RECEnv(
    step_per_episode=10,
    n_items=100,
    n_users=100,
    item_feature_dim=5,
    user_feature_dim=5,
    reward_type="continuous",  # "binary"
    reward_std=0.3,
    obs_std=0.3,
    random_state=12345,
)
```

Specifically, users can define their own `UserModel` as follows.

#### Example of UserModel
```Python
# import recgym modules
from recgym import BaseUserModel
from recgym.types import Action
# import other necessary stuffs
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CustomizedUserModel(BaseUserModel):
    user_feature_dim: int
    item_feature_dim: int
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.coef = self.random_.normal(size=(self.user_feature_dim, self.item_feature_dim))

    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        coefficient = state.T @ self.coef @ item_feature_vector[action]
        state = state + alpha * coefficient * item_feature_vector[action]
        state = state / np.linalg.norm(state, ord=2)
        return state

    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
    ) -> float:
        logit = state.T @ self.coef @ item_feature_vector[action]
        reward = (
            logit if self.reward_type == "continuous" else sigmoid(logit)
        )

        if self.reward_type == "discrete":
            reward = self.random_.binominal(1, p=reward)

        return reward
```

More examples are available at [quickstart/rec/rec_synthetic_customize_env.ipynb](../examples/quickstart/rec/rec_synthetic_customize_env.ipynb). \
The statistics of the environment is also visualized at [quickstart/rec/rec_synthetic_data_collection.ipynb](../examples/quickstart/rec/rec_synthetic_data_collection.ipynb).

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

Any contributions to RECGym are more than welcome!
Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines on how to contribute the project.

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

For any questions about the paper and software, feel free to contact: hk844@cornell.edu

## References

<details>
<summary><strong>Papers </strong>(click to expand)</summary>

1. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

2. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

3. Sarah Dean and Jamie Morgenstern. [Preference Dynamics Under Personalized Recommendations](https://arxiv.org/abs/2205.13026). In *Proceedings of the 23rd ACM Conference on Economics and Computation*, 4503-9150, 2022.

</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is inspired by the following three packages.
- **RecoGym**  -- an RL environment for recommender systems: [[github](https://github.com/criteo-research/reco-gym)] [[paper](https://arxiv.org/abs/1808.00720)]
- **RecSim** -- a configurative RL environment for recommender systems: [[github](https://github.com/google-research/recsim)] [[paper](https://arxiv.org/abs/1909.04847)]
- **AuctionGym** -- an RL environment for online advertising auctions: [[github](https://github.com/amzn/auction-gym)] [[paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym)]
- **FinRL** -- an RL environment for finance: [[github](https://github.com/AI4Finance-Foundation/FinRL)] [[paper](https://arxiv.org/abs/2011.09607)]

</details>

