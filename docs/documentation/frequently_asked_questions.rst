==========
FAQs
==========

SCOPE-RL
~~~~~~~~~~

.. rubric:: Q. xxx environment does not work on SCOPE-RL. How should we fix it?

A. SCOPE-RL is compatible to Open AI Gym and Gymnasium API, specifically for `gym>=0.26.0`, which works as follows.

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

To solve this incompatibility, please use `NewGymAPIWrapper` provided in `scope_rl/utils.py`. It should be used as follows.

.. code-block:: Python

    from scope_rl.utils import NewGymAPIWrapper
    env = NewGymAPIWrapper(env)

.. rubric:: Q. xxx environment does not work on d3rlpy, which is used for model training. How should we fix it? (d3rlpy and SCOPE-RL is compatible to different version of Open AI Gym.)

A. While OFRL is compatible to the latest API of Open AI Gym, d3rlpy is not. Therefore, please use `OldGymAPIWrapper` provided in `scope_rl/utils.py` to enable the use of d3rlpy.

.. code-block:: Python

    from scope_rl.utils import OldGymAPIWrapper
    env = gym.make("xxx_v0")  # compatible to gym>=0.26.2 and SCOPE-RL
    env_ = OldGymAPIWrapper(env)  # compatible to gym<0.26.2 and d3rlpy


.. BasicGym
.. ~~~~~~~~~~

.. RTBGym
.. ~~~~~~~~~~

.. RECGym
.. ~~~~~~~~~~
