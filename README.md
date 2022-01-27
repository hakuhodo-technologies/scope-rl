# OfflineGym: Offline Reinforcement Learning Pipeline for Real World Applications

<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [OfflineGym: Offline Reinforcement Learning Pipeline for Real World Applications](#offlinegym-offline-reinforcement-learning-pipeline-for-real-world-applications)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Synthetic Dataset Generation and Data Preprocessing](#synthetic-dataset-generation-and-data-preprocessing)
  - [Offline Reinforcement Learning](#offline-reinforcement-learning)
  - [Off-Policy Evaluation and Selection](#off-policy-evaluation-and-selection)
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

## Overview

*OfflineGym* is an open-source Python Software for implementing the whole procedure of offline Reinforcement Learning (offline RL), from data collection to offline policy learning, evaluation, and selection. Our software includes a series of modules to implement synthetic dataset generation and dataset preprocessing, Off-Policy Evaluation (OPE) estimators, and also Off-Policy Selection (OPS) methods. The software is also compatible with [d3rlpy](https://github.com/takuseno/d3rlpy), which provides the algorithm implementation of both online and offline RL methods, to streamline the implementation from learning and evaluation in a unified interface. It enables an easy, flexible and reliable experiment in offline RL research on any environment with [OpenAI Gym](https://github.com/st-tech/zr-obp)-like interface (from basic one to practical setup) and also simplify the practical implementation on a variety of customized dataset.

Our software facilitate evaluation and algorithm comparison related to the following research topics:

- **Offline Reinforcement Learning**: Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. OfflineGym enables flexible experiment using customized dataset on a variety of environment collected by various behavior policies.

- **Off-Policy Evaluation**: OPE aims to evaluate the performance of a counterfactual policy using only offline logged data. OfflineGym supports basic implementation of OPE estimators and streamline the experimental procedure to evaluate OPE estimators.

- **Off-Policy Selection**: OPS aims to select the best policy from several candidate policies using offline logged data. OfflineGym supports basic implementation of OPS methods and provide some metrics to evaluate OPS result.

This software is intended for the episodic RL setup. For those aimed for contextual bandits, please also refer to [Open Bandit Pipeline](https://github.com/st-tech/zr-obp). \
To provide an example of conducting customized experiment in a practical setup, we also provide [RTBGym](./rtb_gym), an RL environment for Real-Time Bidding (RTB) under this repository.

### Implementations

*OfflineGym* mainly consists of the following three modules.
- [**dataset module**](./_gym/dataset): This module provides tools to generate synthetic data from any environment with [OpenAI Gym](http://gym.openai.com/)-like interface. It also provides preprocessing tools for the logged data.
- [**policy module**](./_gym/policy) This module provides a wrapper class for [d3rlpy](https://github.com/takuseno/d3rlpy) to enable a flexible data collection.
- [**ope module**](./_gym/ope) This module provides a generic abstract class to implement an OPE estimator and some dominant OPE estimators. It also provides some tools useful for OPS.

<details>
<summary><strong>Behavior Policy </strong>(click to expand)</summary>
<br>

- Discrete
  - Epsilon Greedy
  - Softmax
- Continuous
  - Gaussian
  - Truncated Gaussian

</details>

<details>
<summary><strong>OPE Estimators </strong>(click to expand)</summary>
<br>
- Expected Reward Estimation
  - Direct Method (Fitted Q Evaluation)
  - Trajectory-wise Importance Sampling
  - Step-wise Importance Sampling
  - Doubly Robust
  - Self-Normalized Trajectory-wise Importance Sampling
  - Self-Normalized Step-wise Importance Sampling
  - Self-Normalized Doubly Robust
- Cumulative Distribution Function Estimation

</details>

<details>
<summary><strong>OPS Methods </strong>(click to expand)</summary>
<br>
- Bootstrap
- Hoeffding
- Empirical Bernstein
- T-test
- BayesDICE

</details>

<details>
<summary><strong>Evaluation Metrics of OPS </strong>(click to expand)</summary>
<br>
- Mean Absolute Error
- Mean Squared Error
- Rank Correlation
- Regret

</details>

Note that, in addition to the above OPE and OPS methods, researcher can easily implement compare their own estimators using a generic abstract class.
Practitioners can also use the above implementation with their real-world data to evaluate and choose counterfactual policies.

## Installation

You can install OfflineGym using Python's package manager `pip`.
```
pip install offlinegym
```

You can also install OfflineGym from source.
```bash
git clone https://github.com/negocia-inc/offlinegym
cd offlinegym
python setup.py install
```

OfflineGym supports Python 3.7 or newer. See [pyproject.toml](./pyproject.toml) for other requirements.

## Usage

Here, we provide an example workflow from of offline RL, OPE, and OPS using [RTBGym](./rtb_gym).

### Synthetic Dataset Generation and Data Preprocessing

Let's start by collecting logged data useful for offline RL.

```Python
# implement data collection procedure on the RTBGym environment

# import offlinegym modules
from offlinegym.dataset import SyntheticDataset
from offlinegym.policy import DiscreteEpsilonGreedyHead
# import d3rlpy algorithms
from d3rlpy.algos import DoubleDQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import ConstantEpsilonGreedy
# import rtbgym
from rtbgym import RTBEnv, CustomizedRTBEnv

# (0) Setup environment
env = CustomizedRTBEnv(
    original_env=RTBEnv(random_state=random_state),
    action_type="discrete",
)

# (1) Learn a baseline online policy (using d3rlpy)
# initialize algorithm
ddqn = DoubleDQN()
# train an online policy
ddqn.fit_online(
    env,
    buffer=ReplayBuffer(maxlen=10000, env=env),
    explorer=ConstantEpsilonGreedy(epsilon=0.3),
    n_steps=100000,
    n_steps_per_epoch=1000,
)

# (2) Generate logged dataset
# convert ddqn policy into a stochastic behavior policy
behavior_policy = DiscreteEpsilonGreedyHead(
    ddqn, 
    n_actions=env.action_space.n,
    epsilon=0.3,
    name="ddqn_epsilon_0.3",
    random_state=random_state,
)
# initialize dataset class
dataset = SyntheticDataset(
    env=env,
    behavior_policy=behavior_policy,
    random_state=random_state,
)
# collect logged data using behavior policy
logged_dataset = dataset.obtain_trajectories(n_episodes=10000)
```

### Offline Reinforcement Learning
Now we are ready to learn a new policy only from logged data using d3rlpy.

```Python
# implement offline RL procedure using OfflineGym and d3rlpy

# import d3rlpy algorithms
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL

# (3) Learning a new policy from offline logged data (using d3rlpy)
# convert dataset into d3rlpy's dataset
offlinerl_dataset = MDPDataset(
    observations=logged_dataset["state"],
    actions=logged_dataset["action"],
    rewards=logged_dataset["reward"],
    terminals=logged_dataset["done"],
    episode_terminals=logged_dataset["done"],
    discrete_action=False,
)
# initialize algorithm
cql = DiscreteCQL()
# train an offline policy
cql.fit(
    offlinerl_dataset,
    n_steps=10000,
    scorers={},
)
```

### Off-Policy Evaluation and Selection

Then, let's evaluate the performance of the learned policy using offline logged data. We also compare the estimation results from various OPE estimators, Direct Method (DM), Trajectory-wise Importance Sampling (TIS), Step-wise Importance Sampling (SIS), and Doubly Robust (DR).

```Python
# implement OPE procedure using OfflineGym

# import offlinegym modules
from _gym.ope import CreateOPEInput
from _gym.ope import OffPolicyEvaluation as OPE
from _gym.ope import DiscreteDirectMethod as DM
from _gym.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
from _gym.ope import DiscreteStepWiseImportanceSampling as SIS
from _gym.ope import DiscreteDoublyRobust as DR

# (4) Evaluate the learned policy in an offline manner
# we compare ddqn, cql, and random policy
cql_ = DiscreteEpsilonGreedyHead(
    base_policy=cql, 
    n_actions=env.action_space.n, 
    name="cql", 
    epsilon=0.0, 
    random_state=random_state,
)
ddqn_ = DiscreteEpsilonGreedyHead(
    base_policy=ddqn, 
    n_actions=env.action_space.n, 
    name="ddqn", 
    epsilon=0.0, 
    random_state=random_state,
)
random_ = DiscreteEpsilonGreedyHead(
    base_policy=ddqn, 
    n_actions=env.action_space.n, 
    name="random", 
    epsilon=1.0, 
    random_state=random_state,
)
evaluation_policies = [cql_, ddqn_, random_]
# create input for OPE class
prep = CreateOPEInput(
    logged_dataset=logged_dataset,
    use_base_model=True,  # use model-based prediction
)
input_dict = prep.obtain_whole_inputs(
    evaluation_policies=evaluation_policies,
    env=env,
    n_episodes_on_policy_evaluation=100,
    random_state=random_state,
)
# initialize OPE class
ope = OPE(
    logged_dataset=logged_dataset,
    ope_estimators=[DM(), TIS(), SIS(), DR()],
)
# conduct OPE and visualize the result
ope.visualize_off_policy_estimates(
    input_dict, 
    random_state=random_state, 
    sharey=True,
)
```

A formal quickstart example with RTB is available at [examples/quickstart/rtb_synthetic_discrete.ipynb](./examples/quickstart/rtb_synthetic_discrete.ipynb) (discrete action space) and [examples/quickstart/rtb_synthetic_continuous.ipynb](./examples/quickstart/rtb_synthetic_continuous.ipynb) (continuous action space).

## Citation

If you use our software in your work, please cite our paper:

Haruka Kiyohara, Kosuke Kawakami, Yuta Saito.<br>
**Title**<br>
[link]()

Bibtex:
```
```

## Contribution
Any contributions to OfflineGym are more than welcome!
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for general guidelines how to contribute the project.

## License

This project is licensed under - see [LICENSE](LICENSE) file for details.

## Project Team

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**; Tokyo Institute of Technology)
- Kosuke Kawakami (negocia Inc.)
- [Yuta Saito](https://usaito.github.io/) (Cornell University)

## Contact

For any question about the paper and software, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

## References

<details>
<summary><strong>Papers </strong>(click to expand)</summary>

1. Alina Beygelzimer and John Langford. [The Offset Tree for Learning with Partial Labels](https://arxiv.org/abs/0812.4044). In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 129-138, 2009.

2. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

3. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). In *Statistical Science*, 485-511, 2014.

4. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). In *Proceedings of the 35th International Conference on Machine Learning*, 1861-1870, 2018.

5. Hado van Hasselt, Arthur Guez, and David Silver. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2094-2100, 2015.

6. Nan Jiang and Lihong Li. [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://arxiv.org/abs/1511.03722). In *Proceedings of the 33rd International Conference on Machine Learning*, 652-661, 2016.

7. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*, 3325-3334, 2019.

8. Nathan Kallus and Angela Zhou. [Policy Evaluation and Optimization with Continuous Treatments](https://arxiv.org/abs/1802.06037). In *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics*, 1243-1251, 2019.

9. Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). In *Advances in Neural Information Processing Systems*, 1179-1191, 2020.

10. Hoang Le, Cameron Voloshin, and Yisong Yue. [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738). In *Proceedings of the 36th International Conference on Machine Learning*, 3703-3712, 2019.

11. Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). *arXiv preprint arXiv:2005.01643*, 2020.

12. Doina Precup, Richard S. Sutton, and Satinder P. Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766, 2000.

13. Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). In *Advances in Neural Information Processing Systems*, , 2021.

14. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

15. Alex Strehl, John Langford, Sham Kakade, and Lihong Li. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, 2217-2225, 2010.

16. Adith Swaminathan and Thorsten Joachims. [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). In *Advances in Neural Information Processing Systems*, 3231-3239, 2015.

17. Philip S. Thomas and Emma Brunskill. [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1604.00923). In *Proceedings of the 33rd International Conference on Machine Learning*, 2139-2148, 2016.

</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is strongly inspired by the following two packages.
- **Open Bandit Pipeline**  -- a pipeline implementation of OPE in contextual bandits: [[github](https://github.com/st-tech/zr-obp)] [[paper](https://arxiv.org/abs/2008.07146)]
- **d3rlpy** -- a set of implementations of offline RL algorithms: [[github](https://github.com/takuseno/d3rlpy)] [[paper](https://arxiv.org/abs/2111.03788)]

</details>
