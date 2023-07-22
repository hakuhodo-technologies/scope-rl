# SCOPE-RL: A Python library for offline reinforcement learning, off-policy evaluation, and selection

<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/logo.png" width="100%"/></div>

<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [SCOPE-RL: A Python library for offline reinforcement learning, off-policy evaluation, and selection](#SCOPE-RL-a-python-library-for-offline-reinforcement-learning-off-policy-evaluation-and-selection)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Synthetic Dataset Generation and Data Preprocessing](#synthetic-dataset-generation-and-data-preprocessing)
  - [Offline Reinforcement Learning](#offline-reinforcement-learning)
  - [Basic Off-Policy Evaluation](#basic-off-policy-evaluation)
  - [Advanced Off-Policy Evaluation](#advanced-off-policy-evaluation)
  - [Off-Policy Selection and Evaluation of OPE/OPS](#off-policy-selection-and-evaluation-of-opeops)
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

**Documentation is available [here](https://scope-rl.readthedocs.io/en/latest/)**

**Stable versions are available at [PyPI](https://pypi.org/project/scope-rl/)**

## Overview

*SCOPE-RL* is an open-source Python Software for implementing the end-to-end procedure regarding **offline Reinforcement Learning (offline RL)**, from data collection to offline policy learning, off-policy performance evaluation, and policy selection. Our software includes a series of modules to implement synthetic dataset generation, dataset preprocessing, estimators for Off-Policy Evaluation (OPE), and Off-Policy Selection (OPS) methods.

This software is also compatible with [d3rlpy](https://github.com/takuseno/d3rlpy), which implements a range of online and offline RL methods. SCOPE-RL enables an easy, transparent, and reliable experiment in offline RL research on any environment with [OpenAI Gym](https://gym.openai.com) and [Gymnasium](https://gymnasium.farama.org/)-like interface. It also facilitates the implementation of offline RL in practice on a variety of customized datasets and real-world datasets.

In particular, SCOPE-RL enables and facilitates evaluation and algorithm comparison related to the following research topics:

- **Offline Reinforcement Learning**: Offline RL aims at learning a new policy from only offline logged data collected by a behavior policy. SCOPE-RL enables flexible experiments using customized datasets collected by various behavior policies and environments.

- **Off-Policy Evaluation**: OPE aims at evaluating the performance of a counterfactual policy using only offline logged data. SCOPE-RL supports many OPE estimators and streamlines the experimental procedure to evaluate and compare OPE estimators. Moreover, we also implement advanced OPE methods, such as estimators based on state-action density estimation and cumulative distribution estimation.

- **Off-Policy Selection**: OPS aims at identifying the best-performing policy from a pool of several candidate policies using offline logged data. SCOPE-RL supports some basic OPS methods and provides several metrics to evaluate the OPS accuracy.

This software is inspired by [Open Bandit Pipeline](https://github.com/st-tech/zr-obp), which is a library for OPE in contextual bandits. However, SCOPE-RL also implements a set of OPE estimators and tools to facilitate experiments about OPE for the contextual bandit setup by itself as well as those for RL.

### Implementations

*SCOPE-RL* mainly consists of the following three modules.
- [**dataset module**](./_gym/dataset): This module provides tools to generate synthetic data from any environment on top of [OpenAI Gym](http://gym.openai.com/) and [Gymnasium](https://gymnasium.farama.org/)-like interface. It also provides tools to pre-process the logged data.
- [**policy module**](./_gym/policy): This module provides a wrapper class for [d3rlpy](https://github.com/takuseno/d3rlpy) to enable flexible data collection.
- [**ope module**](./_gym/ope): This module provides a generic abstract class to implement OPE estimators. It also provides some tools useful for performing OPS.

<details>
<summary><strong>Behavior Policy </strong>(click to expand)</summary>

- Discrete
  - Epsilon Greedy
  - Softmax
- Continuous
  - Gaussian
  - Truncated Gaussian

</details>

<details>
<summary><strong>OPE Estimators </strong>(click to expand)</summary>

- Expected Reward Estimation
  - Basic Estimators
    - Direct Method (Fitted Q Evaluation)
    - Trajectory-wise Importance Sampling
    - Per-Decision Importance Sampling
    - Doubly Robust
    - Self-Normalized Trajectory-wise Importance Sampling
    - Self-Normalized Per-Decision Importance Sampling
    - Self-Normalized Doubly Robust
  - State Marginal Estimators
  - State-Action Marginal Estimators
  - Double Reinforcement Learning
  - Weight and Value Learning Methods
    - Augmented Lagrangian Method (BestDICE, DualDICE, GradientDICE, GenDICE, MQL/MWL)
    - Minimax Q-Learning and Weight Learning (MQL/MWL)
- Confidence Interval Estimation
  - Bootstrap
  - Hoeffding
  - (Empirical) Bernstein
  - Student T-test
- Cumulative Distribution Function Estimation
  - Direct Method (Fitted Q Evaluation)
  - Trajectory-wise Importance Sampling
  - Trajectory-wise Doubly Robust
  - Self-Normalized Trajectory-wise Importance Sampling
  - Self-Normalized Trajectory-wise Doubly Robust

</details>

<details>
<summary><strong>OPS Criteria </strong>(click to expand)</summary>

- Policy Value
- Policy Value Lower Bound
- Lower Quartile
- Conditional Value at Risk (CVaR)

</details>

<details>
<summary><strong>Evaluation Metrics of OPS </strong>(click to expand)</summary>

- Mean Squared Error
- Spearman's Rank Correlation Coefficient
- Regret
- Type I and Type II Error Rates
- {Best/Worst/Mean/Std} performances of top-k policies
- Safety violation rate of top-k policies
- SharpeRatio@k

</details>

Note that in addition to the above OPE and OPS methods, researchers can easily implement and compare their own estimators through a generic abstract class implemented in SCOPE-RL. Moreover, practitioners can apply the above methods to their real-world data to evaluate and choose counterfactual policies for their own practical situations.

To provide an example of performing a customized experiment imitating a practical setup, we also provide [RTBGym](./rtbgym) and [RecGym](./recgym), RL environments for Real-Time Bidding (RTB) and Recommender Systems.

## Installation

You can install SCOPE-RL using Python's package manager `pip`.
```
pip install scope-rl
```

You can also install SCOPE-RL from the source.
```bash
git clone https://github.com/hakuhodo-technologies/scope-rl
cd scope-rl
python setup.py install
```

SCOPE-RL supports Python 3.9 or newer. See [requirements.txt](./requirements.txt) for other requirements. Please also refer to issue [#17](https://github.com/hakuhodo-technologies/scope-rl/issues/17) when you encounter some dependency conflicts.

## Usage

Here, we provide an example workflow to perform offline RL, OPE, and OPS using SCOPE-RL on [RTBGym](./rtbgym).

### Synthetic Dataset Generation and Data Preprocessing

Let's start by generating some synthetic logged data useful for performing offline RL.

```Python
# implement a data collection procedure on the RTBGym environment

# import SCOPE-RL modules
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
# import d3rlpy algorithms
from d3rlpy.algos import DoubleDQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import ConstantEpsilonGreedy
# import rtbgym and gym
import rtbgym
import gym
# random state
random_state = 12345

# (0) Setup environment
env = gym.make("RTBEnv-discrete-v0")

# for api compatibility to d3rlpy
from scope_rl.utils import OldGymAPIWrapper
env_ = OldGymAPIWrapper(env)

# (1) Learn a baseline policy in an online environment (using d3rlpy)
# initialize the algorithm
ddqn = DoubleDQN()
# train an online policy
# this takes about 5min to compute
ddqn.fit_online(
    env_,
    buffer=ReplayBuffer(maxlen=10000, env=env_),
    explorer=ConstantEpsilonGreedy(epsilon=0.3),
    n_steps=100000,
    n_steps_per_epoch=1000,
    update_start_step=1000,
)

# (2) Generate a logged dataset
# convert the ddqn policy into a stochastic behavior policy
behavior_policy = EpsilonGreedyHead(
    ddqn,
    n_actions=env.action_space.n,
    epsilon=0.3,
    name="ddqn_epsilon_0.3",
    random_state=random_state,
)
# initialize the dataset class
dataset = SyntheticDataset(
    env=env,
    maximum_episode_steps=env.step_per_episode,
)
# the behavior policy collects some logged data
train_logged_dataset = dataset.obtain_trajectories(
  behavior_policies=behavior_policy,
  n_trajectories=10000,
  random_state=random_state,
)
test_logged_dataset = dataset.obtain_trajectories(
  behavior_policies=behavior_policy,
  n_trajectories=10000,
  random_state=random_state + 1,
)
```

### Offline Reinforcement Learning
We are now ready to learn a new policy (evaluation policy) from the logged data using [d3rlpy](https://github.com/takuseno/d3rlpy).

```Python
# implement an offline RL procedure using SCOPE-RL and d3rlpy

# import d3rlpy algorithms
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL

# (3) Learning a new policy from offline logged data (using d3rlpy)
# convert the logged dataset into d3rlpy's dataset format
offlinerl_dataset = MDPDataset(
    observations=train_logged_dataset["state"],
    actions=train_logged_dataset["action"],
    rewards=train_logged_dataset["reward"],
    terminals=train_logged_dataset["done"],
    episode_terminals=train_logged_dataset["done"],
    discrete_action=True,
)
# initialize the algorithm
cql = DiscreteCQL()
# train an offline policy
cql.fit(
    offlinerl_dataset,
    n_steps=10000,
    scorers={},
)
```

### Basic Off-Policy Evaluation

Then, we evaluate the performance of several evaluation policies (ddqn, cql, and random) using offline logged data collected by the behavior policy. Specifically, we compare the estimation results of various OPE estimators, including Direct Method (DM), Trajectory-wise Importance Sampling (TIS), Per-Decision Importance Sampling (PDIS), and Doubly Robust (DR) below.

```Python
# implement a basic OPE procedure using SCOPE-RL

# import SCOPE-RL modules
from scope_rl.ope import CreateOPEInput
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.discrete import DirectMethod as DM
from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.discrete import PerDecisionImportanceSampling as PDIS
from scope_rl.ope.discrete import DoublyRobust as DR

# (4) Evaluate the learned policy in an offline manner
# we compare ddqn, cql, and random policy
cql_ = EpsilonGreedyHead(
    base_policy=cql,
    n_actions=env.action_space.n,
    name="cql",
    epsilon=0.0,
    random_state=random_state,
)
ddqn_ = EpsilonGreedyHead(
    base_policy=ddqn,
    n_actions=env.action_space.n,
    name="ddqn",
    epsilon=0.0,
    random_state=random_state,
)
random_ = EpsilonGreedyHead(
    base_policy=ddqn,
    n_actions=env.action_space.n,
    name="random",
    epsilon=1.0,
    random_state=random_state,
)
evaluation_policies = [cql_, ddqn_, random_]
# create input for the OPE class
prep = CreateOPEInput(
    logged_dataset=test_logged_dataset,
    use_base_model=True,  # use model-based prediction
)
input_dict = prep.obtain_whole_inputs(
    evaluation_policies=evaluation_policies,
    env=env,
    n_trajectories_on_policy_evaluation=100,
    random_state=random_state,
)
# initialize the OPE class
ope = OPE(
    logged_dataset=test_logged_dataset,
    ope_estimators=[DM(), TIS(), PDIS(), DR()],
)
# perform OPE and visualize the result
ope.visualize_off_policy_estimates(
    input_dict,
    random_state=random_state,
    sharey=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ope_policy_value_basic.png" width="100%"/></div>
<figcaption>
<p align="center">
  Policy Value Estimated by OPE Estimators
</p>
</figcaption>

More formal example implementations with RTBGym are available at [./examples/quickstart/rtb/](./examples/quickstart/rtb). Those with RecGym are also available at [./examples/quickstart/rec/](./examples/quickstart/rec).

### Advanced Off-Policy Evaluation

We can also estimate various statics of the evaluation policy, beyond just its expected performance, including variance and conditional value at risk (CVaR) via estimating the cumulative distribution function (CDF) of the reward under the evaluation policy.

```Python
# implement a cumulative distribution estimation procedure using SCOPE-RL

# import SCOPE-RL modules
from scope_rl.ope import CumulativeDistributionOPE
from scope_rl.ope.discrete import CumulativeDistributionDM as CD_DM
from scope_rl.ope.discrete import CumulativeDistributionTIS as CD_IS
from scope_rl.ope.discrete import CumulativeDistributionTDR as CD_DR
from scope_rl.ope.discrete import CumulativeDistributionSNTIS as CD_SNIS
from scope_rl.ope.discrete import CumulativeDistributionSNTDR as CD_SNDR

# (4) Evaluate the cumulative distribution function of the reward under the evaluation policy (in an offline manner)
# we compare ddqn, cql, and random policy defined from the previous section (i.e., (3) of basic OPE procedure)
# initialize the OPE class
cd_ope = CumulativeDistributionOPE(
    logged_dataset=test_logged_dataset,
    ope_estimators=[
      CD_DM(estimator_name="cd_dm"),
      CD_IS(estimator_name="cd_is"),
      CD_DR(estimator_name="cd_dr"),
      CD_SNIS(estimator_name="cd_snis"),
      CD_SNDR(estimator_name="cd_sndr"),
    ],
)
# estimate the variance
variance_dict = cd_ope.estimate_variance(input_dict)
# estimate the CVaR
cvar_dict = cd_ope.estimate_conditional_value_at_risk(input_dict, alphas=0.3)
# estimate and visualize the cumulative distribution function of the policy performance
cd_ope.visualize_cumulative_distribution_function(input_dict, n_cols=4)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ope_cumulative_distribution_function.png" width="100%"/></div>
<figcaption>
<p align="center">
  Cumulative Distribution Function Estimated by OPE Estimators
</p>
</figcaption>

For more extensive examples, please refer to [quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb).

### Off-Policy Selection and Evaluation of OPE/OPS

We can also select the best-performing policy among a set of candidate policies based on the OPE results using the OPS class. It is also possible to evaluate the reliability of OPE/OPS using various metrics such as mean squaredberror, rank correlation, regret, and type I and type II error rates.

```Python
# perform off-policy selection based on the OPE results

# import SCOPE-RL modules
from scope_rl.ope import OffPolicySelection

# (5) Conduct Off-Policy Selection
# Initialize the OPS class
ops = OffPolicySelection(
    ope=ope,
    cumulative_distribution_ope=cd_ope,
)
# rank the candidate policies by their policy value estimated by (basic) OPE
ranking_dict = ops.select_by_policy_value(input_dict)
# rank the candidate policies by their policy value estimated by cumulative distribution OPE
ranking_dict_ = ops.select_by_policy_value_via_cumulative_distribution_ope(input_dict)
# visualize the top k deployment result
ops.visualize_topk_policy_value_selected_by_standard_ope(
    input_dict=input_dict,
    compared_estimators=["dm", "tis", "pdis", "dr"],
    safety_criteria=1.0,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ops_topk_lower_quartile.png" width="100%"/></div>
<figcaption>
<p align="center">
  Comparison of the Top-k Statistics of 10% Lower Quartile of Policy Value
</p>
</figcaption>

```Python
# (6) Evaluate the OPS/OPE results
# rank the candidate policies by their estimated lower quartile and evaluate the selection results
ranking_df, metric_df = ops.select_by_lower_quartile(
    input_dict,
    alpha=0.3,
    return_metrics=True,
    return_by_dataframe=True,
)
# visualize the OPS results with the ground-truth metrics
ops.visualize_cvar_for_validation(
    input_dict,
    alpha=0.3,
    share_axes=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ops_variance_validation.png" width="100%"/></div>
<figcaption>
<p align="center">
  Validation of Estimated and Ground-truth Variance of Policy Value
</p>
</figcaption>

For more examples, please refer to [quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb) for discrete actions and
[quickstart/rtb/rtb_synthetic_continuous_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_continuous_advanced.ipynb) for continuous actions.

## Citation

If you use our software in your work, please cite our paper:

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection**<br>
[link]() (a preprint coming soon..)

Bibtex:
```
@article{kiyohara2023towards,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year={2023},
}
```

If you use our proposed metric "SharpeRatio@k" in your work, please cite our paper:

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning**<br>
[link]() (a preprint coming soon..)

Bibtex:
```
@article{kiyohara2023towards,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year={2023},
}
```

## Google Group

If you are interested in SCOPE-RL, please follow its updates via the google group:
https://groups.google.com/g/scope-rl


## Contribution
Any contributions to SCOPE-RL are more than welcome!
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for general guidelines on how to contribute to the project.

## License

This project is licensed under Apache 2.0 license - see [LICENSE](LICENSE) file for details.

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

1. Alina Beygelzimer and John Langford. [The Offset Tree for Learning with Partial Labels](https://arxiv.org/abs/0812.4044). In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 129-138, 2009.

2. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

3. Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas. [Universal Off-Policy Evaluation](https://arxiv.org/abs/2104.12820). In *Advances in Neural Information Processing Systems*, 2021.

4. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). In *Statistical Science*, 485-511, 2014.

5. Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang, Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, and Tom Le Paine. [Benchmarks for Deep Off-Policy Evaluation](https://arxiv.org/abs/2103.16596). In *International Conference on Learning Representations*, 2021.

6. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). In *Proceedings of the 35th International Conference on Machine Learning*, 1861-1870, 2018.

7. Josiah P. Hanna, Peter Stone, and Scott Niekum. [Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation](https://arxiv.org/abs/1606.06126). In *Proceedings of the 31th AAAI Conference on Artificial Intelligence*, 2017.

8. Hado van Hasselt, Arthur Guez, and David Silver. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2094-2100, 2015.

9. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment in Contextual Bandits](https://arxiv.org/abs/2104.08977). In *Advances in Neural Information Processing Systems*, 2021.

10. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment for Markov Decision Processes](https://proceedings.mlr.press/v151/huang22b.html). In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics*, 5022-5050, 2022.

11. Nan Jiang and Lihong Li. [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://arxiv.org/abs/1511.03722). In *Proceedings of the 33rd International Conference on Machine Learning*, 652-661, 2016.

13. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*, 3325-3334, 2019.

14. Nathan Kallus and Masatoshi Uehara. [Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes](https://arxiv.org/abs/1908.08526). In *Journal of Machine Learning Research*, 167, 2020.

15. Nathan Kallus and Angela Zhou. [Policy Evaluation and Optimization with Continuous Treatments](https://arxiv.org/abs/1802.06037). In *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics*, 1243-1251, 2019.

16. Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). In *Advances in Neural Information Processing Systems*, 1179-1191, 2020.

17. Vladislav Kurenkov and Sergey Kolesnikov. [Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters](https://arxiv.org/abs/2110.04156). In *Proceedings of the 39th International Conference on Machine Learning*, 11729--11752, 2022.

18. Hoang Le, Cameron Voloshin, and Yisong Yue. [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738). In *Proceedings of the 36th International Conference on Machine Learning*, 3703-3712, 2019.

19. Haanvid Lee, Jongmin Lee, Yunseon Choi, Wonseok Jeon, Byung-Jun Lee, Yung-Kyun Noh, and Kee-Eung Kim. [Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions](https://arxiv.org/abs/2210.13373). In *Advances in Neural Information Processing Systems*, xxxx-xxxx, 2022.

20. Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). *arXiv preprint arXiv:2005.01643*, 2020.

21. Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. [Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation](https://arxiv.org/abs/1810.12429). In *Advances in Neural Information Processing Systems*, 2018.

22. Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. [DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections](https://arxiv.org/abs/1906.04733). In *Advances in Neural Information Processing Systems*, 2019.

23. Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. [AlgaeDICE: Policy Gradient from Arbitrary Experience](https://arxiv.org/abs/1912.02074). *arXiv preprint arXiv:1912.02074*, 2019.

24. Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas. [Hyperparameter Selection for Offline Reinforcement Learning](https://arxiv.org/abs/2007.090550). *arXiv preprint arXiv:2007.09055*, 2020.

25. Doina Precup, Richard S. Sutton, and Satinder P. Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766, 2000.

26. Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, and Esther Luna Colombini. [A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/abs/2203.01387). *arXiv preprint arXiv:2203.01387*, 2022.

27. Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). In *Advances in Neural Information Processing Systems*, 2021.

28. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

29. Alex Strehl, John Langford, Sham Kakade, and Lihong Li. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, 2217-2225, 2010.

30. Adith Swaminathan and Thorsten Joachims. [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). In *Advances in Neural Information Processing Systems*, 3231-3239, 2015.

31. Shengpu Tang and Jenna Wiens. [Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings](https://arxiv.org/abs/2107.11003). In,*Proceedings of the 6th Machine Learning for Healthcare Conference*, 2-35, 2021.

32. Philip S. Thomas and Emma Brunskill. [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1604.00923). In *Proceedings of the 33rd International Conference on Machine Learning*, 2139-2148, 2016.

33. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Off-Policy Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/9541). In *Proceedings of the 9th AAAI Conference on Artificial Intelligence*, 2015.

34. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.html). In *Proceedings of the 32nd International Conference on Machine Learning*, 2380-2388, 2015.

35. Masatoshi Uehara, Jiawei Huang, and Nan Jiang. [Minimax Weight and Q-Function Learning for Off-Policy Evaluation](https://arxiv.org/abs/1910.12809). In *Proceedings of the 37th International Conference on Machine Learning*, 9659--9668, 2020.

36. Masatoshi Uehara, Chengchun Shi, and Nathan Kallus. [A Review of Off-Policy Evaluation in Reinforcement Learning](https://arxiv.org/abs/2212.06355). *arXiv preprint arXiv:2212.06355*, 2022.

37. Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans. [Off-Policy Evaluation via the Regularized Lagrangian](https://arxiv.org/abs/2007.03438). In *Advances in Neural Information Processing Systems*, 6551--6561, 2020.

38. Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum. [SOPE: Spectrum of Off-Policy Estimators](https://arxiv.org/abs/2111.03936). In *Advances in Neural Information Processing Systems*, 18958--18969, 2022.

39. Shangtong Zhang, Bo Liu, and Shimon Whiteson. [GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2001.11113). In *Proceedings of the 37th International Conference on Machine Learning*, 11194--11203, 2020.

40. Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. [GenDICE: Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2002.09072). In *International Conference on Learning Representations*, 2020.

</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is strongly inspired by the following three packages.
- **Open Bandit Pipeline**  -- a pipeline implementation of OPE in contextual bandits: [[github](https://github.com/st-tech/zr-obp)] [[documentation](https://zr-obp.readthedocs.io/en/latest/)] [[paper](https://arxiv.org/abs/2008.07146)]
- **d3rlpy** -- a set of implementations of offline RL algorithms: [[github](https://github.com/takuseno/d3rlpy)] [[documentation](https://d3rlpy.readthedocs.io/en/v0.91/)] [[paper](https://arxiv.org/abs/2111.03788)]
- **Spinning Up** -- an educational resource for learning deep RL: [[github](https://github.com/openai/spinningup)] [[documentation](https://spinningup.openai.com/en/latest/)]

</details>
