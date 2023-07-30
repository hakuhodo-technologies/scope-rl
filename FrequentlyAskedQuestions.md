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

A. A. Both `scope-rl>=0.2.1` and `d3rlpy>=2.0.2` supports compatibility with `gym>=0.26.0` and `gymnasium` environments. The source is available in the `main` branch.

If you want to use the older interface of `d3rlpy`, make sure to use `scope-rl==0.1.3` and `d3rlpy==1.1.1`. Then, please use `OldGymAPIWrapper` provided in `scope_rl/utils.py` to enable the use of d3rlpy. The source is available in the `depreciated` branch.
```Python
from scope_rl.utils import OldGymAPIWrapper
env = gym.make("xxx_v0")  # compatible with gym>=0.26.2 and scope-rl==0.1.3
env_ = OldGymAPIWrapper(env)  # compatible with gym<0.26.2 and d3rlpy==1.1.1
```
