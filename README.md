OfflineGym: Offline Reinforcement Learning Pipeline for Real World Applications

<details>
<summary><strong>Table of Contents</strong></summary>

- [OfflineGym: Offline Reinforcement Learning Pipeline for Real World Applications]
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [(1) Synthetic Dataset Generation and Data Preprocessing]
  - [(2) Offline Reinforcement Learning]
  - [(3) Off-Policy Evaluation and Selection]
- [Citation](#citation)
- [Contribution](#contribution)
- [License](#license)
- [Project Team](#project-team)
- [Contact](#contact)
- [Reference](#reference)

</details>

# Overview

*OfflineGym* is an open-source Python Software for implementing the whole procedure of offline Reinforcement Learning (offline RL), from data collection to offline policy learning, evaluation, and selection. Our software includes a series of modules to implement synthetic dataset generation and dataset preprocessing, Off-Policy Evaluation (OPE) estimators, and also Off-Policy Selection (OPS) methods. The software is also compatible with [d3rlpy](https://github.com/takuseno/d3rlpy), which provides the algorithm implementation of both online and offline RL methods, to streamline the implementation from learning and evaluation in a unified interface. It enables an easy, flexible and reliable experiment in offline RL research on any environment with [OpenAI Gym](https://github.com/st-tech/zr-obp)-like interface (from basic one to practical setup) and also simplify the practical implementation with a variety of custom dataset.

Our software facilitate evaluation and algorithm comparison related to the following research topics:

- **Offline Reinforcement Learning**: Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. OfflineGym enables flexible experiment using customized dataset on a variety of environment collected by various behavior policies.

- **Off-Policy Evaluation**: OPE aims to evaluate the performance of a counterfactual policy using only offline logged data. OfflineGym supports basic implementation of OPE estimators and streamline the experimental procedure to evaluate OPE estimators.

- **Off-Policy Selection**: OPS aims to select the best policy from several candidate policies using offline logged data. OfflineGym supports basic implementation of OPS methods and provide some metrics to evaluate OPS result.

This software is intended for the episodic RL setup. For those aimed for contextual bandits, please also refer to [Open Bandit Pipeline](https://github.com/st-tech/zr-obp).
To provide an example of conducting customized experiment in a practical setup, we also provide [*RTBGym*](./rtb_gym) under this repository.

## Implementations

*OfflineGym* mainly consists of the following three modules.
- [**dataset module**](./_gym/dataset): This module provides tools to generate synthetic data from any environment with [OpenAI Gym](http://gym.openai.com/)-like interface. It also provides preprocessing tools for the logged data.
- [**policy module**](./_gym/policy) This module provides a wrapper class for [d3rlpy](https://github.com/takuseno/d3rlpy) to enable a flexible data collection.
- [**ope module**](./_gym/ope) This module provides a generic abstract class to implement an OPE estimator and some dominant OPE estimators. It also provides some tools useful for OPS.


