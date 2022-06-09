RTBGym; a Python-based configurative simulation environment for Real-Time Bidding (RTB)
===================================

Overview
~~~~~~~~~~
*RTBGym* is an open-source simulation platform for Real-Time Bidding (RTB) of Display Advertising. 
The simulator is particularly intended for reinforcement learning algorithms and follows `OpenAI Gym <https://gym.openai.com>'_ interface. 
We design RTBGym as a configurative environment so that researchers and practitioner can customize the environmental modules including WinningPriceDistribution, ClickThroughRate, and ConversionRate. 

Note that, RTBGym is publicized as a sub-package of `OfflineGym <>`_, which facilitates the implementation of offline reinforcement learning procedure.

Basic Setting
~~~~~~~~~~
In RTB, the objective of the RL agent is to maximize some KPIs (such as numbers of click or conversion) within a episode under the given budget constraints. \
We often aim to achieve this goal by adjusting bidding price function parameter $\alpha$. Specifically, we adjust bid price using $\alpha$ as follows. 

.. math::
    bid_{t,i} = \alpha \cdot r^{\ast}

where :math:`r^{\ast}` denotes predicted or expected reward (KPIs).

We often formulate this RTB problem as the following Constrained Markov Decision Process (CMDP):
    * `timestep`: One episode (a day or a week) consists of several timesteps (24 hours or seven days, for instance).
    * `state`: We observe statistical feedback from environment at each timestep, which include following informations.
        * timestep
        * remaining budget
        * impression level features (budget comsuption rate, cost per mille of impressions, auction winning rate, reward) at previous timestep
        * adjust rate (RL agent's decision making) at previous timestep
    * `action`: Agent chooses adjust rate parameter $\alpha$ to maximize KPIs.
    * `reward`: Total number of clicks or conversions obtained during the timestep.
    * `constraints`: The pre-determined episodic budget should not be exceeded.

Implementation
~~~~~~~~~~

Standard Environment
----------
    * *"RTBEnv-discrete-v0"*: Standard RTB environment with discrete action space.
    * *"RTBEnv-continuous-v0"*: Standard RTB environment with continuous action space.

Custom Environment
----------
    * RTBEnv: The basic configurative environment with continuous action space.
    * CustomizedRTBEnv: The customized environment given action space and reward predictor.

Configurative Modules
----------
    * WinningPriceDistribution: Class to define the winning price distribution of the auction bidding.
    * ClickThroughRate: Class to define the click through rate of users.
    * ConversionRate: Class to define the conversion rate of users.

Note that, users can customize the above modules by following the abstract class. 
We also define the bidding function in the Bidder class and the auction simulation in the Simulator class, respectively.

Quickstart
~~~~~~~~~~


Configuration
~~~~~~~~~~


Citation
~~~~~~~~~~
If you use our pipeline in your work, please cite our paper below.

```
@article{kiyohara2021accelerating,
  title={Accelerating Offline Reinforcement Learning Application in Real-Time Bidding and Recommendation: Potential Use of Simulation},
  author={Kiyohara, Haruka and Kawakami, Kosuke and Saito, Yuta},
  journal={arXiv preprint arXiv:2109.08331},
  year={2021}
}
```

Contact
~~~~~~~~~~
For any question about the paper and pipeline, feel free to contact: kiyohara.h.aa@m.titech.ac.jp

Contribution
~~~~~~~~~~
Any contributions to OfflineGym are more than welcome!
Please refer to `CONTRIBUTING.md <>`_ for general guidelines how to contribute to the project.