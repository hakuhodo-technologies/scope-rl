# Benchmarking SCOPE RL

This directory includes the code to replicate the benchmark experiment done in the following paper.

Haruka Kiyohara, Ren Kishimoto, Kousuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning**<br>
[https://arxiv.org/abs/](https://arxiv.org/abs/)


If you find this code useful in your research then please cite:
```
@article{,
  title={Risk-Return Assessments of Off-Policy Evaluation in Reinforcement Learning},
  author={Haruka Kiyohara, Ren Kishimoto, Kousuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito},
  journal={arXiv preprint arXiv:},
  year={2023}
}
```

## Setting
We use continuous control benchmarks such as Reacher, InvertedPendulum, Hopper, and Swimmer from Gym-Mujuco and discrete controls such as CartPole, MountainCar, and Acrobot from Gym-Classic Control.

## Evaluating Off-Policy Estimators
In the benchmark experiment, we evaluate the estimation performance of the following OPE estimators.

- Direct Method (DM)
- Per-Decision Importance Sampling (PDIS) 
- Doubly Robust (DR)
- Marginal Importance Sampling (MIS)
- Marginal Doubly Robust (MDR)

<!-- See Section 4.2 of [our paper](https://arxiv.org/abs/) for the details of these estimators. -->

## Dependencies
This repository supports Python 3.7 or newer.

- numpy==1.22.4
- pandas==1.5.3
- seaborn==0.12.2
- matplotlib==3.7.1
- gym==0.26.2
- d3rlpy==1.1.1
- mujoco==2.3.5
- hydra-core==1.3.2

<!-- 
### Selecting Env
If you use CartPoleEnv, you change conf/config.yaml as shown below
```bash
defaults:
  - setting: cartpole
  - base_model_config: cartpole
  - visualize: cartpole


hydra: 
  run: 
    dir: ./
  sweep:
    dir: ./
    subdir: ./
``` -->

## Running the code
To conduct the synthetic experiment, prepare and run the following commands.


(i) setting Env\
If you use CartPoleEnv, you change conf/config.yaml as shown below
```bash
defaults:
  - setting: CartPole
  - base_model_config: CartPole
  - visualize: CartPole


hydra: 
  run: 
    dir: ./
  sweep:
    dir: ./
    subdir: ./
```

(ii) learning a behavior policy and candidate policies.
```bash
python policy_learning.py 
```

(iii) conducting OPE and OPS
```bash
python ope.py
```
Once the code is finished executing, you can find the results in the `./logs/results/` directory. 

### Visualize the results
To visualize the results, run the following commands.
Make sure that you have executed the above two experiments (by running `python policy_learning.py` and `python ope.py`) before visualizing the results.
```bash
python visualize.py
```