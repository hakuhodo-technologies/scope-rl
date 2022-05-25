# OfflineGym: A pipeline for offline reinforcement learning research and applications
<details>
<summary><strong>Table of Contents </strong>(click to expand)</summary>

- [OfflineGym: A pipeline for offline reinforcement learning research and applications](#offlinegym-a-pipeline-for-offline-reinforcement-learning-research-and-applications)
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Synthetic Dataset Generation and Data Preprocessing](#synthetic-dataset-generation-and-data-preprocessing)
  - [Offline Reinforcement Learning](#offline-reinforcement-learning)
  - [Basic Off-Policy Evaluation](#basic-off-policy-evaluation)
  - [Advanced Off-Policy Evaluation](#advanced-off-policy-evaluation)
    - [Cumulative Distribution and Statistics Estimation](#cumulative-distribution-and-statistics-estimation)
    - [Distributionally Robust Metrics Estimation](#distributionally-robust-metrics-estimation)
  - [Off-Policy Selection](#off-policy-selection)
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

## Overview

*OfflineGym* is an open-source Python Software for implementing the whole procedure of offline Reinforcement Learning (offline RL), from data collection to offline policy learning, evaluation, and selection. Our software includes a series of modules to implement synthetic dataset generation and dataset preprocessing, estimators for Off-Policy Evaluation (OPE), and Off-Policy Selection (OPS) methods. 

This software is also compatible with [d3rlpy](https://github.com/takuseno/d3rlpy), which provides the algorithm implementation of both online and offline RL methods, to streamline the implementation from learning and evaluation in an unified interface. OfflineGym enables an easy, flexible and reliable experiment in offline RL research on any environment with [OpenAI Gym](https://gym.openai.com)-like interface (from basic one to practical setup) and also simplify the practical implementation on a variety of customized dataset.

Our software facilitate evaluation and algorithm comparison related to the following research topics:

- **Offline Reinforcement Learning**: Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. OfflineGym enables flexible experiment using customized dataset on a variety of environment collected by various behavior policies.

- **Off-Policy Evaluation**: OPE aims to evaluate the performance of a counterfactual policy using only offline logged data. OfflineGym supports basic implementation of OPE estimators and streamline the experimental procedure to evaluate OPE estimators. Moreover, we also implement advanced OPE, such as cumulative distribution estimators.

- **Off-Policy Selection**: OPS aims to select the best policy from several candidate policies using offline logged data. OfflineGym supports basic implementation of OPS methods and provide some metrics to evaluate OPS result.

This software is intended for the episodic RL setup. For those aimed for contextual bandits, please also refer to [Open Bandit Pipeline](https://github.com/st-tech/zr-obp).

### Implementations

*OfflineGym* mainly consists of the following three modules.
- [**dataset module**](./_gym/dataset): This module provides tools to generate synthetic data from any environment with [OpenAI Gym](http://gym.openai.com/)-like interface. It also provides preprocessing tools for the logged data.
- [**policy module**](./_gym/policy): This module provides a wrapper class for [d3rlpy](https://github.com/takuseno/d3rlpy) to enable a flexible data collection.
- [**ope module**](./_gym/ope): This module provides a generic abstract class to implement an OPE estimator and some dominant OPE estimators. It also provides some tools useful for OPS.

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
  - Direct Method (Fitted Q Evaluation)
  - Trajectory-wise Importance Sampling
  - Per-Decision Importance Sampling
  - Doubly Robust
  - Self-Normalized Trajectory-wise Importance Sampling
  - Self-Normalized Per-Decision Importance Sampling
  - Self-Normalized Doubly Robust
- Confidence Interval Estimation
  - Bootstrap
  - Hoeffding
  - (Empirical) Bernstein
  - Student T-test
- Cumulative Distribution Function and Statistics Estimation
  - Direct Method (Fitted Q Evaluation)
  - Importance Sampling
  - Doubly Robust
  - Self-Normalized Importance Sampling
  - Self-Normalized Doubly Robust
- Distributionally Robust Metric Estimation (work in progress)
  - Importance Sampling
  - Self-Normalized Importance Sampling
  - Doubly Robust

</details>

<details>
<summary><strong>OPS Criterion </strong>(click to expand)</summary>

- Policy Value
- Policy Value Lower Bound
- Lower Quartile
- Conditional Value at Risk (CVaR)
- Distributionally Robust Worst Case Policy Value

</details>

<details>
<summary><strong>Evaluation Metrics of OPS </strong>(click to expand)</summary>

- Mean Squared Error
- Spearman's Rank Correlation Coefficient
- Regret
- Type I and Type II Error Rates

</details>

Note that, in addition to the above OPE and OPS methods, researcher can easily implement compare their own estimators using a generic abstract class.
Practitioners can also use the above implementation with their real-world data to evaluate and choose counterfactual policies.

To provide an example of conducting customized experiment in a practical setup, we also provide [RTBGym](./rtbgym), an RL environment for Real-Time Bidding (RTB) under this repository.

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

Here, we provide an example workflow from of offline RL, OPE, and OPS using [RTBGym](./rtbgym).

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
Now we are ready to learn a new policy only from logged data using [d3rlpy](https://github.com/takuseno/d3rlpy).

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
    discrete_action=True,
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

### Basic Off-Policy Evaluation

Then, we evaluate the performance of the learned policy using offline logged data. We also compare the estimation results from various OPE estimators, Direct Method (DM), Trajectory-wise Importance Sampling (TIS), Per-Decision Importance Sampling (PDIS), and Doubly Robust (DR).

```Python
# implement basic OPE procedure using OfflineGym

# import offlinegym modules
from _gym.ope import CreateOPEInput
from _gym.ope import OffPolicyEvaluation as OPE
from _gym.ope import DiscreteDirectMethod as DM
from _gym.ope import DiscreteTrajectoryWiseImportanceSampling as TIS
from _gym.ope import DiscretePerDecisionImportanceSampling as PDIS
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
    ope_estimators=[DM(), TIS(), PDIS(), DR()],
)
# conduct OPE and visualize the result
ope.visualize_off_policy_estimates(
    input_dict, 
    random_state=random_state, 
    sharey=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/negocia/offlinegym/master/images/ope_policy_value_basic.png" width="60%"/></div>
<figcaption>
<p align="center">
  Policy Value Estimated by OPE Estimators
</p>
</figcaption>

A formal quickstart example with RTBGym is available at [quickstart/rtb_synthetic_discrete_basic.ipynb](./examples/quickstart/rtb_synthetic_discrete_basic.ipynb) (discrete action space) and [quickstart/rtb_synthetic_continuous_basic.ipynb](./examples/quickstart/rtb_synthetic_continuous_basic.ipynb) (continuous action space).

### Advanced Off-Policy Evaluation

We can also estimate various performance statics including variance and conditional value at risk (CVaR) by using cumulative distributional OPE estimators.

```Python
# implement advanced OPE procedure (i.e., cumulative distribution estimation) using OfflineGym

# import offlinegym modules
from offlinegym.ope import DiscreteCumulativeDistributionalOffPolicyEvaluation as CumulativeDistributionalOPE
from offlinegym.ope import DiscreteCumulativeDistributionalDirectMethod as CD_DM
from offlinegym.ope import DiscreteCumulativeDistributionalImportanceSampling as CD_IS
from offlinegym.ope import DiscreteCumulativeDistributionalDoublyRobust as CD_DR
from offlinegym.ope import DiscreteCumulativeDistributionalSelfNormalizedImportanceSampling as CD_SNIS
from offlinegym.ope import DiscreteCumulativeDistributionalSelfNormalizedDoublyRobust as CD_SNDR

# (4) Evaluate the learned policy using cumulative distribution function (in an offline manner)
# we compare ddqn, cql, and random policy defined in the previous section (i.e., (3) of basic OPE procedure)
# initialize OPE class
cd_ope = CumulativeDistributionalOPE(
    logged_dataset=logged_dataset,
    ope_estimators=[CD_DM(), CD_IS(), CD_DR(), CD_SNIS(), CD_SNDR()],
    use_observations_as_reward_scale=True,
)
# estimate variance
variance_dict = cd_ope.estimate_variance(input_dict)
# estimate CVaR
cvar_dict = cd_ope.estimate_conditional_value_at_risk(input_dict, alphas=0.3)
# estimate and visualize cumulative distribution function
cd_ope.visualize_cumulative_distribution_function(input_dict, n_cols=4)
```
<div align="center"><img src="https://raw.githubusercontent.com/negocia/offlinegym/master/images/ope_cumulative_distribution_function.png" width="60%"/></div>
<figcaption>
<p align="center">
  Cumulative Distribution Function Estimated by OPE Estimators
</p>
</figcaption>

For more examples, please refer to [quickstart/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb_synthetic_discrete_advanced.ipynb).

### Off-Policy Selection and Evaluation of OPE/OPS

Finally, we select policy based on the OPE result using OPS class. We also evaluate the reliability of OPE/OPS using various metrics
including mean-squared-error, rank correlation, regret, and type I and type II error rates.

```Python
# conduct off-policy selection based on the OPE results

# import offlinegym modules
from offlinegym.ope import OffPolicySelection

# (5) Conduct Off-Policy Selection
# Initialize OPS class
ops = OffPolicySelection(
    ope=ope,
    cumulative_distributional_ope=cd_ope,
)
# rank candidate policy by policy value estimated by (basic) OPE
ranking_dict = ops.select_by_policy_value(input_dict)
# rank candidate policy by policy value estimated by cumulative distributional OPE
ranking_dict_ = ops.select_by_policy_value_via_cumulative_distributional_ope(input_dict)

# (6) Evaluate OPS/OPE results
# rank candidate policy by estimated lower quartile and evaluate the selection results
ranking_df, metric_df = ops.select_by_lower_quartile(
    input_dict,
    alpha=0.3,
    return_metrics=True,
    return_by_dataframe=True,
)
# visualize the OPS results with the ground-truth metrics
ops.visualize_lower_quartile_for_validation(
    input_dict,
    alpha=0.3,
    share_axes=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/negocia/offlinegym/master/images/ops_cvar_validation.png" width="60%"/></div>
<figcaption>
<p align="center">
  Comparison of Estimated and Ground-truth Lower Quartile of Policy Value
</p>
</figcaption>

For more examples, please refer to [quickstart/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb_synthetic_discrete_advanced.ipynb).

## Citation

If you use our software in your work, please cite our paper:

Haruka Kiyohara, Kosuke Kawakami, Yuta Saito.<br>
**Title**<br>
[link]()

Bibtex:
```
```

### Additional Relevant Papers

For your information, our previous workshop paper (which has been appeared in RecSys'21 SimuRec workshop) may also be helpful.

Haruka Kiyohara, Kosuke Kawakami, Yuta Saito.<br>
**Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation**<br>
[https://arxiv.org/abs/2109.08331](https://arxiv.org/abs/2109.08331)

## Contribution
Any contributions to OfflineGym are more than welcome!
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for general guidelines how to contribute the project.

## License

This project is licensed under Apache 2.0 license - see [LICENSE](LICENSE) file for details.

## Project Team

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**; Tokyo Institute of Technology)
- Kosuke Kawakami (negocia Inc.)
- [Yuta Saito](https://usait0.com/en/) (Cornell University)

## Contact

For any question about the paper and software, feel free to contact: kiyohara.h.aa@m.titech.ac.jp.

## References

<details>
<summary><strong>Papers </strong>(click to expand)</summary>

1. Alina Beygelzimer and John Langford. [The Offset Tree for Learning with Partial Labels](https://arxiv.org/abs/0812.4044). In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 129-138, 2009.

2. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

3. Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas. [Universal Off-Policy Evaluation](https://arxiv.org/abs/2104.12820). In *Advances in Neural Information Processing Systems*, 2021.

4. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). In *Statistical Science*, 485-511, 2014.

5. Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang, Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, and Tom Le Paine. [Benchmarks for Deep Off-Policy Evaluation](https://arxiv.org/abs/2103.16596). In *International Conference on Learning Representations*, 2021.

6. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). In *Proceedings of the 35th International Conference on Machine Learning*, 1861-1870, 2018.

7. Josiah P. Hanna, Peter Stone, and Scott Niekum. [Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation](https://arxiv.org/abs/1606.06126). In *Proceedings of the 16th International Conference on Autonomous Agents and Multi-agent Systems*, 2017.

8. Hado van Hasselt, Arthur Guez, and David Silver. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2094-2100, 2015.

9. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment in Contextual Bandits](https://arxiv.org/abs/2104.08977). In *Advances in Neural Information Processing Systems*, 2021.

10. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment for Markov Decision Processes](https://proceedings.mlr.press/v151/huang22b.html). In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics*, 5022-5050, 2022.

11. Nan Jiang and Lihong Li. [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://arxiv.org/abs/1511.03722). In *Proceedings of the 33rd International Conference on Machine Learning*, 652-661, 2016.

12. Nathan Kallus, Xiaojie Mao, Kaiwen Wang, and Zhengyuan Zhou. [Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning](https://arxiv.org/abs/2202.09667). In *Proceedings of the 39th International Conference on Machine Learning*, ,2022.

13. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*, 3325-3334, 2019.

14. Nathan Kallus and Angela Zhou. [Policy Evaluation and Optimization with Continuous Treatments](https://arxiv.org/abs/1802.06037). In *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics*, 1243-1251, 2019.

15. Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). In *Advances in Neural Information Processing Systems*, 1179-1191, 2020.

16. Hoang Le, Cameron Voloshin, and Yisong Yue. [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738). In *Proceedings of the 36th International Conference on Machine Learning*, 3703-3712, 2019.

17. Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). *arXiv preprint arXiv:2005.01643*, 2020.

18. Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas. [Hyperparameter Selection for Offline Reinforcement Learning](https://arxiv.org/abs/2007.090550). *arXiv preprint arXiv:2007.09055*, 2020.

19. Doina Precup, Richard S. Sutton, and Satinder P. Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766, 2000.

20. Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). In *Advances in Neural Information Processing Systems*, , 2021.

21. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

22. Nian Si, Fan Zhang, Zhengyuan Zhou, and Jose Blanchet. [Distributional Robust Batch Contextual Bandits](https://arxiv.org/abs/2006.05630). In *Proceedings of the 37th International Conference on Machine Learning*, 8884-8894, 2020.

23. Alex Strehl, John Langford, Sham Kakade, and Lihong Li. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, 2217-2225, 2010.

24. Adith Swaminathan and Thorsten Joachims. [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). In *Advances in Neural Information Processing Systems*, 3231-3239, 2015.

25. Shengpu Tang and Jenna Wiens. [Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings](https://arxiv.org/abs/2107.11003). In,*Proceedings of the 6th Machine Learning for Healthcare Conference*, 2-35, 2021.

26. Philip S. Thomas and Emma Brunskill. [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1604.00923). In *Proceedings of the 33rd International Conference on Machine Learning*, 2139-2148, 2016.

27. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Off-Policy Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/9541). In *Proceedings of the 9th AAAI Conference on Artificial Intelligence*, 2015.

28. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.html). In *Proceedings of the 32nd International Conference on Machine Learning*, 2380-2388, 2015.

</details>

<details>
<summary><strong>Projects </strong>(click to expand)</summary>

This project is strongly inspired by the following three packages.
- **Open Bandit Pipeline**  -- a pipeline implementation of OPE in contextual bandits: [[github](https://github.com/st-tech/zr-obp)] [[documentation](https://zr-obp.readthedocs.io/en/latest/)] [[paper](https://arxiv.org/abs/2008.07146)]
- **d3rlpy** -- a set of implementations of offline RL algorithms: [[github](https://github.com/takuseno/d3rlpy)] [[documentation](https://d3rlpy.readthedocs.io/en/v0.91/)] [[paper](https://arxiv.org/abs/2111.03788)]
- **Spinning Up** -- an educational resource for learning deepl RL: [[github](https://github.com/openai/spinningup)] [[documentation](https://spinningup.openai.com/en/latest/)]

</details>
