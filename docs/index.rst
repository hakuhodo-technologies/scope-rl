.. OfflineGym documentation master file, created by
   sphinx-quickstart on Thu Jan 20 15:25:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OfflineGym; a Python library for offline reinforcement learning and off-policy evaluation
===================================

Overview
~~~~~~~~~~
*OfflineGym* is an open-source Python library for offline Reinforcement Learning (RL) and Off-Policy Evaluation (OPE).
This library aims to facilitate the an easy, flexible and reliable experiment in offline RL research and to provide a streamlined implementation also for practitioners.
OfflineGym includes a series of modules to implement synthetic dataset generation and dataset preprocessing, a variety of OPE estimators, and also Off-Policy Selection (OPS) methods. 

OfflineGym is applicable to any RL environment with [OpenAI Gym](https://github.com/st-tech/zr-obp)-like interface.
The library is also compatible with [d3rlpy](https://github.com/takuseno/d3rlpy), which provides the algorithm implementation of both online and offline RL methods.

Our software facilitate implementation, evaluation and algorithm comparison related to the following research topics:

- **Offline Reinforcement Learning**: Offline RL aims to learn a new policy from only offline logged data collected by a behavior policy. 
OfflineGym enables flexible experiment using customized dataset on diverse environments collected by various behavior policies.

- **Off-Policy Evaluation**: OPE aims to evaluate the performance of a counterfactual policy using only offline logged data. 
OfflineGym supports basic implementation of OPE estimators and streamline the experimental procedure to evaluate OPE estimators.

- **Off-Policy Selection**: OPS aims to select the best policy from several candidate policies using offline logged data. 
OfflineGym supports basic implementation of OPS methods and provide some metrics to evaluate OPS result.

This website contains pages with example implementations that demonstrates the usage of this library.
The package reference page contains the full reference documentation for the currently implemented modules.

implementation
~~~~~~~~~~

Data Collection Policy and Offline RL
----------
OfflineGym override `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation for the base algorithm. 
We provide a wrapper class for transforming the policy into a stochastic policy as follows.
   * Epsilon Greedy (discrete)
   * Softmax (discrete)
   * Gaussian (continuous)
   * Truncated Gaussian (continuous)


Discrete / Continuous OPE
----------
   * Direct Method (DM) :cite:`beygelzimer2009offset`
   * Trajectory-wise Importance Sampling (TIS) :cite:`precup2000eligibility`
   * Step-wise Importance Sampling (SIS) :cite:`precup2000eligibility`
   * Doubly Robust (DR) :cite:`jiang2016doubly` :cite:`thomas2016data`
   * Self-Normalized Trajectory-wise Importance Sampling (SNTIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
   * Self-Normalized Step-wise Importance Sampling (SNSIS) :cite:`precup2000eligibility` :cite:`kallus2020optimal`
   * Self-Normalized Doubly Robust (SNDR) :cite:`jiang2016doubly` :cite:`thomas2016data` :cite:`kallus2020optimal`

Off-Policy Selection
----------

Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

```
@article{
   title={},
   author={},
   journal={},
   year={},
}
```

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

Contribution
~~~~~~~~~~
Any contributions to OfflineGym are more than welcome!
Please refer to `CONTRIBUTING.md <>`_ for general guidelines how to contribute to the project.

Table of Contents
~~~~~~~~~~

.. toctree::
   :maxdepth: 3
   :caption: Introduction:

   about
   related

.. toctree::
   :maxdepth: 3
   :caption: Online Reinforcement Learning (online RL):

   online_rl

.. toctree::
   :maxdepth: 3
   :caption: Offline Reinforcement Learning (offline RL):

   offline_rl
   wrapper

.. toctree::
   :maxdepth: 3
   :caption: Off-Policy Evaluation (OPE):

   ope
   estimators
   evaluation_of_ope

.. toctree::
   :maxdepth: 3
   :caption: Off-Policy Selection (OPS):

   ops
   evaluation_of_ops

.. toctree::
   :maxdepth: 3
   :caption: Getting Started:

   installation
   quickstart

.. toctree::
   :maxdepth: 3
   :caption: Package Reference:

.. toctree::
   :caption: Others:

   Github <>
   LISENSE <>
   references

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
