# Benchmarking OPE estimators with SharpRatio@k

This directory includes the code to replicate the benchmark experiment done in the following paper.

Haruka Kiyohara, Ren Kishimoto, Kousuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning**<br>
[link]() (a preprint coming soon..)

If you find this code useful in your research then please cite:
```
@article{kiyohara2023towards,
  title={Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning},
  author={Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  journal = {A github repository},
  pages = {xxx--xxx},
  year = {2023},
}
```

## Setting
We use continuous control benchmarks such as Reacher, InvertedPendulum, Hopper, and Swimmer from Gym-Mujuco and discrete controls such as CartPole, MountainCar, and Acrobot from Gym-Classic Control.

## Compared Off-Policy Estimators
In the benchmark experiment, we evaluate the estimation performance of the following OPE estimators.

- Direct Method (DM)
- Per-Decision Importance Sampling (PDIS) 
- Doubly Robust (DR)
- Marginal Importance Sampling (MIS)
- Marginal Doubly Robust (MDR)

See Section 4.2 and Appendix C.1 of our paper or the package documentation for the details of these estimators.

## Dependencies
This repository supports Python 3.9 or newer.

- numpy==1.24.3
- scipy==1.10.1
- scikit-learn==1.0.2
- torch==2.0.0
- pandas==2.0.2
- seaborn==0.12.2
- matplotlib==3.7.1
- gym==0.26.2
- gymnasium==0.28.1
- mujoco==2.3.5
- d3rlpy==1.1.1
- hydra-core==1.3.2
- scope-rl==0.1.0

## Running the code
To conduct the synthetic experiment, run the following commands. Note that, make sure that the path is connected to `scope-rl` and `scope-rl/experiments` directories.

(i) learning a behavior policy and candidate policies.
```bash
python policy_learning.py setting=CartPole base_model_config=CartPole
```

(ii) performing and evaluating OPE and OPS
```bash
python ope.py setting=CartPole base_model_config=CartPole
```

Once the code is finished executing, you can find the results in the `./logs/results/` directory. 

### Visualize the results
To visualize the results, run the following commands.
Make sure that you have executed the above two experiments (by running `python policy_learning.py` and `python ope.py`) before visualizing the results.

```bash
python visualize.py setting=CartPole
```

The figures are stored in the `./logs/results/` directory. For the summary and results, please refer to Section 4.3 and Appendix A.2 of our paper.
