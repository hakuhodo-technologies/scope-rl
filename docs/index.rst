.. RTB-Gym documentation master file, created by
   sphinx-quickstart on Thu Jan 20 15:25:38 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RTBGym: Offline Reinforcement Learning Pipeline for Real World Applications
===================================

Overview
~~~~~~~~~~


implementation
~~~~~~~~~~

Data Collection Policy and Offline RL
----------
We override `d3rlpy <https://github.com/takuseno/d3rlpy>`_'s implementation for the base algorithm, and provide a wrapper for deriving stochastic policy as follows.
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

RTB Env
----------
Standardized RTBEnv is available at OpenAI Gym. We also provide configurative environment that users can customize
   * Winning Price Distribution
   * Click Through Rate
   * Conversion Rate

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
   :caption: Real Time Bidding (RTB):

   rtb
   simulation

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
