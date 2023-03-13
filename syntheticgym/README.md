# SyntheticGym: A configurative reinforcement learning environment for synthetic simulation
<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [SyntheticGym: A reinforcement learning environment for synthetic simulation](#SyntheticGym-a-reinforcement-learning-environment-for-synthetic-simulation)
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

*SyntheticGym* is an open-source simulation platform for synthetic simulation, which is written in Python. The simulator is particularly intended for reinforcement learning algorithms and follows [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface. We design SyntheticGym as a configurative environment so that researchers and practitioner can customize the environmental modules including `StateTransition` and `RewardFunction`

Note that, SyntheticGym is publicized under [OfflineGym](../) repository, which facilitates the implementation of offline reinforcement learning procedure.

### Basic Setting

In synthetic simulation, the objective of the RL agent is to maximize reward. \
We often formulate this synthetic simulation problem as the following (Partially Observable) Markov Decision Process ((PO)MDP):
- `state`: 
   - When the true state is unobservable, you can gain observation instead of state.
- `action`:  
   - Indicating which action to present to the context.
- `reward`:
   - Either binary or continuous.

### Implementation

SyntheticGym provides a syntheticmmender environment.
- `"SyntheticEnv-v0"`: Standard syntheticmmender environment.

SyntheticGym consists of the following a environments.
- [SyntheticEnv](./envs/synthetic.py#L17): The basic configurative environment.

SyntheticGym is configurative about the following a module.
- [StateTransition](./envs/simulator/function.py#L14): Class to define the state transition of the synthetic simulation.
- [RewardFunction](./envs/simulator/function.py#L93): Class to define the reward function of the synthetic simulation.

Note that, users can customize the above modules by following the [abstract class](./envs/simulator/base.py).

## Installation
SyntheticGym can be installed as a part of [OfflineGym](../) using Python's package manager `pip`.
```
pip install offlinegym
```

You can also install from source.
```bash
git clone https://github.com/negocia-inc/offlinegym
cd offlinegym
python setup.py install
```

## Usage

We provide an example usage of the standard and customized environment. \
The online/offlline RL and Off-Policy Evaluation examples are provides in [OfflineGym's README](../README.md).

### Standard SyntheticEnv

Our standard SyntheticEnv is available from `gym.make()`, following the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

```Python
# import SyntheticGym and gym
import syntheticgym
import gym

# (1) standard environment 
env = gym.make('SyntheticEnv-v0')
```

The basic interaction is performed using only four lines of code as follows.

```Python
obs, info = env.reset(), False
while not done:
    action = agent.sample_action_online(obs)
    obs, reward, done, truncated, info = env.step(action)
```

Let's visualize case with uniform random policy .

```Python
# import from other libraries
from offlinegym.policy import DiscreteEpsilonGreedyHead
from d3rlpy.algos import RandomPolicy as DiscreteRandomPolicy

# define a random agent
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
  Transition of the Reward during a Single Episode
</p>
</figcaption>

Note that, while we use [OfflineGym](../README.md) and [d3rlpy](https://github.com/takuseno/d3rlpy) here, SyntheticGym is compatible with any other libraries working on the [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface.

### Customized SyntheticEnv

Next, we describe how to customize the environment by instantiating the environment.

<details>
<summary>List of environmental configurations: (click to expand)</summary>

- `StateTransition`: State transition of the synthetic simulation.
- `RewardFunction`: Reward function of the synthetic simulation.
- `state_dim`: Dimensions of state.
- `action_type`: action type (i.e., continuous / discrete).
- `n_actions`: Number of actions. Applicable only when reward_type is "discrete".
- `action_dim`: Dimensions of the action context.
- `action_context`: Feature vectors that characterizes each action. Applicable only when reward_type is "discrete".
- `reward_type`: Reward type (i.e., continuous / binary).
- `reward_std`: Standard deviation of the reward distribution. Applicable only when reward_type is "continuous".
- `obs_std`: Standard deviation of the observation distribution.
- `step_per_episode`: Number of timesteps in an episode.
- `random_state` : Random state

</details>

```Python
from syntheticgym import SyntheticEnv
env = SyntheticEnv(
        StateTransition = StateTransition
        RewardFunction = RewardFunction
        state_dim = 5, #each state has 5 dimensional features
        action_type = "continuous", #we use continuous action
        action_dim = 3,  #each action has 10 dimensional features
        action_context = None,  #determine action_context from n_actions and action_dim in SyntheticEnv
        reward_type = "continuous", #we use continuous reward
        reward_std = 0.0,
        obs_std = 0.0, #not add noise to the observation
        step_per_episode = 10,
        random_state = 12345,
)
```

Specifically, users can define their own `StateTransition` as follows.

#### Example of StateTransition
```Python
# import syntheticgym modules
from syntheticgym import BaseStateTransition
from sytheticgym.types import Action
# import other necessary stuffs
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class StateTransition(BaseStateTransition):
    state_dim: int = 5
    action_type: str = "continuous",  # "binary"
    action_dim: int = 3
    action_context: Optional[np.ndarray] = (None,)
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)

        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim))
        self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim))


    def step(
        self,
        state: np.ndarray,
        action: Action,
    ) -> np.ndarray:

        if self.action_type == "continuous":
            state = self.state_coef @ state / self.state_dim +  self.action_coef @ action / self.action_dim + (self.state_action_coef @ action / self.action_dim).T @ state / self.state_dim
        
        elif self.action_type == "discrete":
            state = self.state_coef @ state / self.state_dim + self.action_coef @ self.action_context[action] / self.action_dim +  (self.state_action_coef @ self.action_context[action] / self.action_dim).T @ state / self.state_dim
            
        state = state / np.linalg.norm(state, ord=2)



```
Specifically, users can define their own `RewardFunction` as follows.

#### Example of RewardFunction
```Python
# import syntheticgym modules
from syntheticgym import BaseRewardFunction
from sytheticgym.types import Action
# import other necessary stuffs
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class RewardFunction(BaseRewardFunction):
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    state_dim: int = 5
    action_type: str = "continuous",  # "discrete"
    action_dim: int = 3
    action_context: Optional[np.ndarray] = (None,)
    random_state: Optional[int] = None

    def __post_init__(self):
        check_scalar(
            self.reward_std,
            name="reward_std",
            target_type=float,
        )

        if self.reward_type not in ["continuous", "binary"]:
            raise ValueError(
                f'reward_type must be either "continuous" or "binary", but {self.reward_type} is given'
            )

        self.random_ = check_random_state(self.random_state)

        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, ))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.action_dim, ))
        self.state_action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim))

    def sample(
        self,
        state: np.ndarray,
        action: Action,
    ) -> float:
        if self.action_type == "continuous":
            reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ action / self.action_dim + state.T @ (self.state_action_coef @ action / self.action_dim) / self.state_dim
        
        elif self.action_type == "discrete":
            reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ self.action_context[action] / self.action_dim + state.T @ (self.state_action_coef @ self.action_context[action] / self.action_dim) / self.state_dim 

        if self.reward_type == "continuous":
            reward = reward + self.random_.normal(loc=0.0, scale=self.reward_std)

        return reward


```

<!-- More examples are available at [quickstart/synthetic_synthetic_customize_env.ipynb](./examples/quickstart/synthetic_synthetic_customize_env.ipynb). \
The statistics of the environment is also visualized at [quickstart/synthetic_synthetic_data_collection.ipynb](./examples/quickstart/synthetic_synthetic_data_collection.ipynb). -->


## Contribution
Any contributions to SyntheticGym are more than welcome!
Please refer to [CONTRIBUTING.md](../CONTRIBUTING.md) for general guidelines how to contribute the project.

## License

This project is licensed under Apache 2.0 license - see [LICENSE](../LICENSE) file for details.

## Project Team

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**; Tokyo Institute of Technology)
- Kosuke Kawakami (negocia Inc.)
- [Yuta Saito](https://usait0.com/en/) (Cornell University)

## Contact

For any question about the paper and software, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

## References

<details>
<summary><strong>Papers </strong>(click to expand)</summary>

1. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

2. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.


</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is inspired by the following three packages.
- **syntheticGym**  -- an RL environment for synthetic simulations: [[github](https://github.com/criteo-research/synthetic-gym)] [[paper](https://arxiv.org/abs/1808.00720)]
- **syntheticSim** -- a configurative RL environment for synthetic simulations: [[github](https://github.com/google-research/syntheticsim)] [[paper](https://arxiv.org/abs/1909.04847)]
- **AuctionGym** -- an RL environment for online advertising auctions: [[github](https://github.com/amzn/auction-gym)] [[paper](https://www.amazon.science/publications/learning-to-bid-with-auctiongym)]
- **FinRL** -- an RL environment for finance: [[github](https://github.com/AI4Finance-Foundation/FinRL)] [[paper](https://arxiv.org/abs/2011.09607)]

</details>

