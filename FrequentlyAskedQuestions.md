# Frequently Asked Questions

Q. xxx environment does not work on SCOPE-RL. How should we fix it?

A. SCOPE-RL is compatible with OpenAI Gym API, specifically for `gym>=0.26.0`, which works as follows.
```Python
obs, info = env.reset(), False
while not done:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
```

In contrast, your environment may use the following older interface.
```Python
obs = env.reset(), False
while not done:
    action = agent.act(obs)
    obs, reward, done, info = env.step(action)
```

To solve this incompatibility, please use `NewGymAPIWrapper` provided in `scope_rl/utils.py`. It should be used as follows.
```Python
from scope_rl.utils import NewGymAPIWrapper
env = NewGymAPIWrapper(env)
```

Q. xxx environment does not work on d3rlpy, which is used for model training. How should we fix it? (d3rlpy and SCOPE-RL are compatible with different version of OpenAI Gym.)

A. While SCOPE-RL is compatible with the latest API of OpenAI Gym, d3rlpy is not. Therefore, please use `OldGymAPIWrapper` provided in `scope_rl/utils.py` to enable the use of d3rlpy.
```Python
from scope_rl.utils import OldGymAPIWrapper
env = gym.make("xxx_v0")  # compatible with gym>=0.26.2 and SCOPE-RL
env_ = OldGymAPIWrapper(env)  # compatible with gym<0.26.2 and d3rlpy
```
