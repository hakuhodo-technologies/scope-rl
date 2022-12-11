==========
FAQs
==========

OFRL
~~~~~~~~~~

.. rubric:: Q. xxx environment does not work on OFRL. How should we fix it?

A. OFRL is compatible to Open AI Gym and Gymnasium API, specifically for `gym>=0.26.0`, which works as follows. 

.. code-block:: Python

    obs, info = env.reset(), False
    while not done:
        action = agent.act(obs)
        obs, reward, done, truncated, info = env.step(action)

In contrast, your environment may use the following older interface.

.. code-block:: Python

    obs = env.reset(), False
    while not done:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

To solve this incompatibility, please use `NewGymAPIWrapper` provided in `ofrl/utils.py`. It should be used as follows.

.. code-block:: Python

    from ofrl.utils import NewGymAPIWrapper
    env = NewGymAPIWrapper(env)

.. rubric:: Q. xxx environment does not work on d3rlpy, which is used for model training. How should we fix it? (d3rlpy and OFRL is compatible to different version of Open AI Gym.)

A. While OFRL is compatible to the latest API of Open AI Gym, d3rlpy is not. Therefore, please use `OldGymAPIWrapper` provided in `ofrl/utils.py` to make the environment work for d3rlpy.

.. code-block:: Python

    from ofrl.utils import OldGymAPIWrapper
    env = gym.make("xxx_v0")  # compatible to gym>=0.26.2 and OFRL
    env_ = OldGymAPIWrapper(env)  # compatible to gym<0.26.2 and d3rlpy


RTBGym
~~~~~~~~~~