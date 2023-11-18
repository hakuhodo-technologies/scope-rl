# SCOPE-RL: A Python library for offline reinforcement learning, off-policy evaluation, and selection

<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/logo.png" width="100%"/></div>

[![pypi](https://img.shields.io/pypi/v/scope-rl.svg)](https://pypi.python.org/pypi/scope-rl)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org)
[![Downloads](https://pepy.tech/badge/scope-rl)](https://pepy.tech/project/scope-rl)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hakuhodo-technologies/scope-rl)](https://github.com/hakuhodo-technologies/scope-rl/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/hakuhodo-technologies/scope-rl)](https://github.com/hakuhodo-technologies/scope-rl/graphs/commit-activity)
[![Documentation Status](https://readthedocs.org/projects/scope-rl/badge/?version=latest)](https://scope-rl.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv-23xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/23xx.xxxxx)

<details>
<summary><strong>目次</strong>(クリックして展開)</summary>

- [SCOPE-RL: A Python library for offline reinforcement learning, off-policy evaluation, and selection](#SCOPE-RL-a-python-library-for-offline-reinforcement-learning-off-policy-evaluation-and-selection)
- [概要](#概要)
- [インストール](#インストール)
- [用法](#usage)
  - [人工データ生成と前処理](#synthetic-dataset-generation-and-data-preprocessing)
  - [オフライン強化学習](#offline-reinforcement-learning)
  - [標準的なオフ方策評価](#basic-off-policy-evaluation)
  - [発展的なオフ方策評価](#advanced-off-policy-evaluation)
  - [オフ方策選択とOPE/OPSの評価](#off-policy-selection-and-evaluation-of-opeops)
- [引用](#citation)
- [貢献](#contribution)
- [ライセンス](#license)
- [プロジェクトチーム](#project-team)
- [連絡先](#contact)
- [参考文献](#reference)

</details>

**ドキュメンテーション [documentation](https://scope-rl.readthedocs.io/en/latest/)**

**PyPIで最新版が利用可能 [PyPI](https://pypi.org/project/scope-rl/)**

## 概要
SCOPE-RL は，データ収集からオフ方策学習，オフ方策性能評価，方策選択をend-to-endで実装するためのオープンソースのPythonソフトウェアです．私たちのソフトウェアには，人工データ生成，データの前処理，オフ方策評価 (off-policy evaluation; OPE) の推定量，オフ方策選択 (off-policy selection; OPS) 手法を実装するための一連のモジュールが含まれています．

このソフトウェアは，オンラインおよびオフラインの強化学習 (reinforment learning; RL) 手法を実装する[d3rlpy](https://github.com/takuseno/d3rlpy)とも互換性があります．また，SCOPE-RLは [OpenAI Gym](https://gym.openai.com) や [Gymnasium](https://gymnasium.farama.org/) のインターフェースに基づく環境であればどのような設定の環境でも使用できます．さらに，様々なカスタマイズされたデータや実データに対して，オフライン強化学習の実践的な実装や，透明性や信頼性の高い実験を容易にします．


特に，SCOPE-RLは以下の研究トピックに関連する評価とアルゴリズム比較を簡単に行えます：

- **オフライン強化学習**：オフライン強化学習は，挙動方策によって収集されたオフラインのログデータのみから新しい方策を学習することを目的としています．SCOPE-RLは，様々な挙動方策と環境によって収集されたデータによる柔軟な実験を可能にします．

- **オフ方策評価(OPE)**：オフ方策評価は，挙動方策により集められたオフラインのログデータのみを使用して（挙動方策とは異なる）新たな方策の性能を評価することを目的とします．SCOPE-RLは多くのオフ方策推定量の実装可能にする抽象クラスや、推定量を評価し比較するための実験手順を実装しています．また、SCOPE-RLが実装し公開している発展的なオフ方策評価手法には、状態-行動密度推定や累積分布推定に基づく推定量なども含まれます．

- **オフ方策選択(OPS)**：オフ方策選択は，オフラインのログデータを使用して，いくつかの候補方策の中から最も性能の良い方策を特定することを目的とします．SCOPE-RLは様々な方策選択の基準を実装するだけでなく，方策選択の結果を評価するためのいくつかの指標を提供します．

このソフトウェアは，文脈付きバンディットにおけるオフ方策評価のためのライブラリである[Open Bandit Pipeline](https://github.com/st-tech/zr-obp)を参考にしています．SCOPE-RLはオフ方策評価の実験を容易にするための一連のオフ方策推定量と手法を実装しており，より強化学習の設定に適したものになっています．


### 実装

*SCOPE-RL* は主に以下の3つのモジュールから構成されています．

- [**dataset module**](./_gym/dataset): このモジュールは，[OpenAI Gym](http://gym.openai.com/) や[Gymnasium](https://gymnasium.farama.org/)のようなインターフェイスに基づく任意の環境から人工データを生成するためのツールを提供します．また，ログデータの前処理を行うためのツールも提供します．
- [**policy module**](./_gym/policy): このモジュールはd3rlpyのwrapperクラスを提供し，柔軟なデータ収集を可能にします．
- [**ope module**](./_gym/ope): このモジュールは，OPE推定量を実装するための汎用的な抽象クラスを提供します．また，OPSを実行するためのいくつかの便利なツールも提供します．

<details>
<summary><strong>挙動方策</strong>(クリックして展開)</summary>

- Discrete
  - Epsilon Greedy
  - Softmax
- Continuous
  - Gaussian
  - Truncated Gaussian

</details>

<details>
<summary><strong>OPE推定量</strong>(クリックして展開)</summary>

- Expected Reward Estimation
  - Basic Estimators
    - Direct Method (Fitted Q Evaluation)
    - Trajectory-wise Importance Sampling
    - Per-Decision Importance Sampling
    - Doubly Robust
    - Self-Normalized Trajectory-wise Importance Sampling
    - Self-Normalized Per-Decision Importance Sampling
    - Self-Normalized Doubly Robust
  - State Marginal Estimators
  - State-Action Marginal Estimators
  - Double Reinforcement Learning
  - Weight and Value Learning Methods
    - Augmented Lagrangian Method (BestDICE, DualDICE, GradientDICE, GenDICE, MQL/MWL)
    - Minimax Q-Learning and Weight Learning (MQL/MWL)
- Confidence Interval Estimation
  - Bootstrap
  - Hoeffding
  - (Empirical) Bernstein
  - Student T-test
- Cumulative Distribution Function Estimation
  - Direct Method (Fitted Q Evaluation)
  - Trajectory-wise Importance Sampling
  - Trajectory-wise Doubly Robust
  - Self-Normalized Trajectory-wise Importance Sampling
  - Self-Normalized Trajectory-wise Doubly Robust

</details>

<details>
<summary><strong>OPSの基準</strong>(クリックして展開)</summary>

- Policy Value
- Policy Value Lower Bound
- Lower Quartile
- Conditional Value at Risk (CVaR)

</details>

<details>
<summary><strong>OPSの評価指標</strong>(クリックして展開)</summary>

- Mean Squared Error
- Spearman's Rank Correlation Coefficient
- Regret
- Type I and Type II Error Rates
- {Best/Worst/Mean/Std} performances of top-k policies
- Safety violation rate of top-k policies
- SharpeRatio@k

</details>

OPEおよびOPS手法に加えて，研究者はSCOPE-RLに実装された一般的な抽象クラスを通じて自分の推定量を簡単に実装し，比較することができます．さらに，実務家はこれらの方法を実データに適用し，自身の実際の状況に合った反実的方策を評価し，選択することができます．

実用的なセットアップを模倣し，カスタマイズされた実験を行う例として，Real-Time Bidding(RTB)と推薦システム用の強化学習環境である[RTBGym](./rtbgym)と[RecGym](./recgym)も提供します



## インストール

pythonの`pip`を利用してSCOPE-RLをインストールできます.
```
pip install scope-rl
```

またコードからSCOPE-RLをインストールすることもできます.
```bash
git clone https://github.com/hakuhodo-technologies/scope-rl
cd scope-rl
python setup.py install
```

SCOPE-RLはPython 3.9以降をサポートしています．その他の要件については[requirements.txt](./requirements.txt)を参照してください．依存関係の競合が発生した場合は，[#17](https://github.com/hakuhodo-technologies/scope-rl/issues/17)も参照してください．

## 用法

ここでは，[RTBGym](./rtbgym)を使用してSCOPE-RLでオフライン強化学習，OPE，OPSを実行するための例を紹介します．

### 人工データの生成と前処理

まず，オフライン強化学習を実行するために役立ついくつかの人工ログデータを生成します．

```Python
# RTBGym環境でのデータ収集手順を実装する

# SCOPE-RLモジュールをインポートする
from scope_rl.dataset import SyntheticDataset
from scope_rl.policy import EpsilonGreedyHead
# d3rlpyのアルゴリズムをインポートする
from d3rlpy.algos import DoubleDQNConfig
from d3rlpy.dataset import create_fifo_replay_buffer
from d3rlpy.algos import ConstantEpsilonGreedy
# rtbgymとgymをインポートする
import rtbgym
import gym
import torch
# 乱数の状態
random_state = 12345
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# (0) 環境のセットアップ
env = gym.make("RTBEnv-discrete-v0")

# (1) オンライン環境で基本方策を学習する(d3rlpyを使用)
# アルゴリズムを初期化する
ddqn = DoubleDQNConfig().create(device=device)
# オンライン挙動方策を訓練する
# 約5分かかる
ddqn.fit_online(
    env,
    buffer=create_fifo_replay_buffer(limit=10000, env=env),
    explorer=ConstantEpsilonGreedy(epsilon=0.3),
    n_steps=100000,
    n_steps_per_epoch=1000,
    update_start_step=1000,
)

# (2) ログデータを生成する
# ddqn方策を確率的な挙動方策に変換する
behavior_policy = EpsilonGreedyHead(
    ddqn,
    n_actions=env.action_space.n,
    epsilon=0.3,
    name="ddqn_epsilon_0.3",
    random_state=random_state,
)
# データクラスを初期化する
dataset = SyntheticDataset(
    env=env,
    max_episode_steps=env.step_per_episode,
)
# 挙動方策がいくつかのログデータを収集する
train_logged_dataset = dataset.obtain_episodes(
  behavior_policies=behavior_policy,
  n_trajectories=10000,
  random_state=random_state,
)
test_logged_dataset = dataset.obtain_episodes(
  behavior_policies=behavior_policy,
  n_trajectories=10000,
  random_state=random_state + 1,
)

```

### オフライン強化学習
ここまででログデータを使用して新しい方策(評価方策)を学習する準備が整いました．[d3rlpy](https://github.com/takuseno/d3rlpy)を使用します．

```Python
# SCOPE-RLとd3rlpyを使用してオフラインRL手順を実装する

# d3rlpyのアルゴリズムをインポートする
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQLConfig

# (3) オフラインログデータから新しい方策を学習する(d3rlpyを使用)
# ログデータをd3rlpyのデータ形式に変換する
offlinerl_dataset = MDPDataset(
    observations=train_logged_dataset["state"],
    actions=train_logged_dataset["action"],
    rewards=train_logged_dataset["reward"],
    terminals=train_logged_dataset["done"],
)
# アルゴリズムを初期化する
cql = DiscreteCQLConfig().create(device=device)
# オフライン方策を学習する
cql.fit(
    offlinerl_dataset,
    n_steps=10000,
)

```

### 標準的なオフ方策評価

次に，挙動方策によって収集されたオフラインのログデータを使用して，いくつかの評価方策(ddqn，cql，random)のパフォーマンスを評価します．具体的には，Direct Method (DM)，Trajectory-wise Importance Sampling (TIS)，Per-Decision Importance Sampling (PDIS)，Doubly Robust (DR)を含む様々なOPE推定量の推定結果を比較します．

```Python
# SCOPE-RLを使用して基本的なOPE手順を実装する

# SCOPE-RLモジュールをインポート
from scope_rl.ope import CreateOPEInput
from scope_rl.ope import OffPolicyEvaluation as OPE
from scope_rl.ope.discrete import DirectMethod as DM
from scope_rl.ope.discrete import TrajectoryWiseImportanceSampling as TIS
from scope_rl.ope.discrete import PerDecisionImportanceSampling as PDIS
from scope_rl.ope.discrete import DoublyRobust as DR

# (4) 学習した方策をオフラインで評価する
# ここではddqn，cql，randomを比較する
cql_ = EpsilonGreedyHead(
    base_policy=cql,
    n_actions=env.action_space.n,
    name="cql",
    epsilon=0.0,
    random_state=random_state,
)
ddqn_ = EpsilonGreedyHead(
    base_policy=ddqn,
    n_actions=env.action_space.n,
    name="ddqn",
    epsilon=0.0,
    random_state=random_state,
)
random_ = EpsilonGreedyHead(
    base_policy=ddqn,
    n_actions=env.action_space.n,
    name="random",
    epsilon=1.0,
    random_state=random_state,
)
evaluation_policies = [cql_, ddqn_, random_]
# OPEクラス用の入力を作成する
prep = CreateOPEInput(
    env=env,
)
input_dict = prep.obtain_whole_inputs(
    logged_dataset=test_logged_dataset,
    evaluation_policies=evaluation_policies,
    require_value_prediction=True,
    n_trajectories_on_policy_evaluation=100,
    random_state=random_state,
)
# OPEクラスを初期化する
ope = OPE(
    logged_dataset=test_logged_dataset,
    ope_estimators=[DM(), TIS(), PDIS(), DR()],
)
# OPEを実行し，結果を視覚化する
ope.visualize_off_policy_estimates(
    input_dict,
    random_state=random_state,
    sharey=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ope_policy_value_basic.png" width="100%"/></div>
<figcaption>
<p align="center">
  Policy Value Estimated by OPE Estimators
</p>
</figcaption>

より正式なRTBGymを使用した実装の例は，[./examples/quickstart/rtb/](./examples/quickstart/rtb)で利用できます．RecGymを使用した例も[./examples/quickstart/rec/](./examples/quickstart/rec)で利用可能です．


### 発展的なオフ方策評価

評価方策の期待性能だけでなく，その分散やconditional value at risk (CVaR)など，様々な統計を推定することもできます．これは，評価方策のもとでの報酬の累積分布関数(CDF)を推定することで行います．

```Python
# SCOPE-RLを使用して累積分布推定手順を実装する

# SCOPE-RLモジュールをインポートする
from scope_rl.ope import CumulativeDistributionOPE
from scope_rl.ope.discrete import CumulativeDistributionDM as CD_DM
from scope_rl.ope.discrete import CumulativeDistributionTIS as CD_IS
from scope_rl.ope.discrete import CumulativeDistributionTDR as CD_DR
from scope_rl.ope.discrete import CumulativeDistributionSNTIS as CD_SNIS
from scope_rl.ope.discrete import CumulativeDistributionSNTDR as CD_SNDR

# (4) 評価方策の下での報酬の累積分布関数をオフラインで評価する
# 前のセクションで定義されたddqn，cql，randomを比較する(基本的なOPE手順の(3))
# OPEクラスを初期化する
cd_ope = CumulativeDistributionOPE(
    logged_dataset=test_logged_dataset,
    ope_estimators=[
      CD_DM(estimator_name="cd_dm"),
      CD_IS(estimator_name="cd_is"),
      CD_DR(estimator_name="cd_dr"),
      CD_SNIS(estimator_name="cd_snis"),
      CD_SNDR(estimator_name="cd_sndr"),
    ],
)
# 分散を推定する
variance_dict = cd_ope.estimate_variance(input_dict)
# CVaRを推定する
cvar_dict = cd_ope.estimate_conditional_value_at_risk(input_dict, alphas=0.3)
# 方策性能の累積分布関数を推定し，可視化する
cd_ope.visualize_cumulative_distribution_function(input_dict, n_cols=4)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ope_cumulative_distribution_function.png" width="100%"/></div>
<figcaption>
<p align="center">
  OPE推定量による累積分布関数の推定
</p>
</figcaption>


より詳細な例については[quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb)を参照してください.

### オフ方策選択とOPE/OPSの評価

OPSクラスを使用して，OPEの結果に基づき，候補方策の中から最も性能の高い方策を選択することもできます．mean squared error，rank correlation，regret，type I や type IIの誤差率など，様々な指標を使用してOPE/OPSの信頼性を評価することも可能です．

```Python
# OPEの結果に基づいてオフ方策選択を行う

# SCOPE-RLモジュールをインポートする
from scope_rl.ope import OffPolicySelection

# (5) オフ方策選択を実施する
# OPSクラスを初期化する
ops = OffPolicySelection(
    ope=ope,
    cumulative_distribution_ope=cd_ope,
)
# (標準的な) OPEによって推定された方策価値に基づいて候補方策をランク付けする
ranking_dict = ops.select_by_policy_value(input_dict)

# 累積分布OPEによって推定された方策価値に基づいて候補方策をランク付けする
ranking_dict_ = ops.select_by_policy_value_via_cumulative_distribution_ope(input_dict)

# top-kのデプロイ結果を視覚化する
ops.visualize_topk_policy_value_selected_by_standard_ope(
    input_dict=input_dict,
    compared_estimators=["dm", "tis", "pdis", "dr"],
    relative_safety_criteria=1.0,
)

```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ops_topk_lower_quartile.png" width="100%"/></div>
<figcaption>
<p align="center">
  top-kの統計量の比較 (方策価値の下位四分位10%)
</p>
</figcaption>

```Python
# (6) OPS/OPEの結果を評価する
# 推定された下位四分位数によって候補方策をランク付けし，選択結果を評価する
ranking_df, metric_df = ops.select_by_lower_quartile(
    input_dict,
    alpha=0.3,
    return_metrics=True,
    return_by_dataframe=True,
)
# 真の性能を利用し，OPS結果を視覚化する
ops.visualize_conditional_value_at_risk_for_validation(
    input_dict,
    alpha=0.3,
    share_axes=True,
)
```
<div align="center"><img src="https://raw.githubusercontent.com/hakuhodo-technologies/scope-rl/main/images/ops_variance_validation.png" width="100%"/></div>
<figcaption>
<p align="center">
  方策価値の推定した分散と真の分散
</p>
</figcaption>

より具体的には離散行動空間は[quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_discrete_advanced.ipynb)，連続行動空間は
[quickstart/rtb/rtb_synthetic_continuous_advanced.ipynb](./examples/quickstart/rtb/rtb_synthetic_continuous_advanced.ipynb) を参照してください

## 引用

ソフトウェアを使用する場合は，以下の論文を引用してください．

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection**<br>
[link]() (a preprint coming soon..)

Bibtex:
```
@article{kiyohara2023towards,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year={2023},
}
```

我々が提案した指標"SharpeRatio@k"を使用する場合は，以下の論文を引用してください．

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning**<br>
[link]() (a preprint coming soon..)

Bibtex:
```
@article{kiyohara2023towards,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {Towards Assessing and Benchmarking Risk-Return Tradeoff of Off-Policy Evaluation in Reinforcement Learning},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year={2023},
}
```

## Google グループ
SCOPE-RLに興味がある場合は，Googleグループを通じて更新情報をフォローしてください
https://groups.google.com/g/scope-rl


## 貢献
SCOPE-RLへの貢献も歓迎しています！
プロジェクトへの貢献方法については， [CONTRIBUTING.md](./CONTRIBUTING.md)を参照してください．


## ライセンス
このプロジェクトはApache 2.0ライセンスのもとでライセンスされています - 詳細については[LICENSE](LICENSE)ファイルをご覧ください．


## プロジェクトチーム

- [Haruka Kiyohara](https://sites.google.com/view/harukakiyohara) (**Main Contributor**)
- 岸本 廉 (Ren Kishimoto) (東京工業大学)
- 川上 孝介 (Kosuke Kawakami) (博報堂テクノロジーズ)
- 小林 健 (Ken Kobayashi) (東京工業大学)
- 中田 和秀 (Kazuhide Nakata) (東京工業大学)
- [齋藤 優太 (Yuta Saito)](https://usait0.com/en/) (コーネル大学)

## 連絡先

論文やソフトウェアに関する質問がある場合は，hk844@cornell.eduまでお気軽にお問い合わせください．


## 参考文献

<details>
<summary><strong>論文 </strong>(クリックして展開)</summary>

1. Alina Beygelzimer and John Langford. [The Offset Tree for Learning with Partial Labels](https://arxiv.org/abs/0812.4044). In *Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 129-138, 2009.

2. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

3. Yash Chandak, Scott Niekum, Bruno Castro da Silva, Erik Learned-Miller, Emma Brunskill, and Philip S. Thomas. [Universal Off-Policy Evaluation](https://arxiv.org/abs/2104.12820). In *Advances in Neural Information Processing Systems*, 2021.

4. Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li. [Doubly Robust Policy Evaluation and Optimization](https://arxiv.org/abs/1503.02834). In *Statistical Science*, 485-511, 2014.

5. Justin Fu, Mohammad Norouzi, Ofir Nachum, George Tucker, Ziyu Wang, Alexander Novikov, Mengjiao Yang, Michael R. Zhang, Yutian Chen, Aviral Kumar, Cosmin Paduraru, Sergey Levine, and Tom Le Paine. [Benchmarks for Deep Off-Policy Evaluation](https://arxiv.org/abs/2103.16596). In *International Conference on Learning Representations*, 2021.

6. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). In *Proceedings of the 35th International Conference on Machine Learning*, 1861-1870, 2018.

7. Josiah P. Hanna, Peter Stone, and Scott Niekum. [Bootstrapping with Models: Confidence Intervals for Off-Policy Evaluation](https://arxiv.org/abs/1606.06126). In *Proceedings of the 31th AAAI Conference on Artificial Intelligence*, 2017.

8. Hado van Hasselt, Arthur Guez, and David Silver. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). In *Proceedings of the AAAI Conference on Artificial Intelligence*, 2094-2100, 2015.

9. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment in Contextual Bandits](https://arxiv.org/abs/2104.08977). In *Advances in Neural Information Processing Systems*, 2021.

10. Audrey Huang, Liu Leqi, Zachary C. Lipton, and Kamyar Azizzadenesheli. [Off-Policy Risk Assessment for Markov Decision Processes](https://proceedings.mlr.press/v151/huang22b.html). In *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics*, 5022-5050, 2022.

11. Nan Jiang and Lihong Li. [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://arxiv.org/abs/1511.03722). In *Proceedings of the 33rd International Conference on Machine Learning*, 652-661, 2016.

13. Nathan Kallus and Masatoshi Uehara. [Intrinsically Efficient, Stable, and Bounded Off-Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1906.03735). In *Advances in Neural Information Processing Systems*, 3325-3334, 2019.

14. Nathan Kallus and Masatoshi Uehara. [Double Reinforcement Learning for Efficient Off-Policy Evaluation in Markov Decision Processes](https://arxiv.org/abs/1908.08526). In *Journal of Machine Learning Research*, 167, 2020.

15. Nathan Kallus and Angela Zhou. [Policy Evaluation and Optimization with Continuous Treatments](https://arxiv.org/abs/1802.06037). In *Proceedings of the 21st International Conference on Artificial Intelligence and Statistics*, 1243-1251, 2019.

16. Aviral Kumar, Aurick Zhou, George Tucker, and Sergey Levine. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779). In *Advances in Neural Information Processing Systems*, 1179-1191, 2020.

17. Vladislav Kurenkov and Sergey Kolesnikov. [Showing Your Offline Reinforcement Learning Work: Online Evaluation Budget Matters](https://arxiv.org/abs/2110.04156). In *Proceedings of the 39th International Conference on Machine Learning*, 11729--11752, 2022.

18. Hoang Le, Cameron Voloshin, and Yisong Yue. [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738). In *Proceedings of the 36th International Conference on Machine Learning*, 3703-3712, 2019.

19. Haanvid Lee, Jongmin Lee, Yunseon Choi, Wonseok Jeon, Byung-Jun Lee, Yung-Kyun Noh, and Kee-Eung Kim. [Local Metric Learning for Off-Policy Evaluation in Contextual Bandits with Continuous Actions](https://arxiv.org/abs/2210.13373). In *Advances in Neural Information Processing Systems*, xxxx-xxxx, 2022.

20. Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643). *arXiv preprint arXiv:2005.01643*, 2020.

21. Qiang Liu, Lihong Li, Ziyang Tang, and Dengyong Zhou. [Breaking the Curse of Horizon: Infinite-Horizon Off-Policy Estimation](https://arxiv.org/abs/1810.12429). In *Advances in Neural Information Processing Systems*, 2018.

22. Ofir Nachum, Yinlam Chow, Bo Dai, and Lihong Li. [DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections](https://arxiv.org/abs/1906.04733). In *Advances in Neural Information Processing Systems*, 2019.

23. Ofir Nachum, Bo Dai, Ilya Kostrikov, Yinlam Chow, Lihong Li, and Dale Schuurmans. [AlgaeDICE: Policy Gradient from Arbitrary Experience](https://arxiv.org/abs/1912.02074). *arXiv preprint arXiv:1912.02074*, 2019.

24. Tom Le Paine, Cosmin Paduraru, Andrea Michi, Caglar Gulcehre, Konrad Zolna, Alexander Novikov, Ziyu Wang, and Nando de Freitas. [Hyperparameter Selection for Offline Reinforcement Learning](https://arxiv.org/abs/2007.090550). *arXiv preprint arXiv:2007.09055*, 2020.

25. Doina Precup, Richard S. Sutton, and Satinder P. Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs). In *Proceedings of the 17th International Conference on Machine Learning*, 759–766, 2000.

26. Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, and Esther Luna Colombini. [A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/abs/2203.01387). *arXiv preprint arXiv:2203.01387*, 2022.

27. Yuta Saito, Shunsuke Aihara, Megumi Matsutani, and Yusuke Narita. [Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation](https://arxiv.org/abs/2008.07146). In *Advances in Neural Information Processing Systems*, 2021.

28. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.

29. Alex Strehl, John Langford, Sham Kakade, and Lihong Li. [Learning from Logged Implicit Exploration Data](https://arxiv.org/abs/1003.0120). In *Advances in Neural Information Processing Systems*, 2217-2225, 2010.

30. Adith Swaminathan and Thorsten Joachims. [The Self-Normalized Estimator for Counterfactual Learning](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). In *Advances in Neural Information Processing Systems*, 3231-3239, 2015.

31. Shengpu Tang and Jenna Wiens. [Model Selection for Offline Reinforcement Learning: Practical Considerations for Healthcare Settings](https://arxiv.org/abs/2107.11003). In,*Proceedings of the 6th Machine Learning for Healthcare Conference*, 2-35, 2021.

32. Philip S. Thomas and Emma Brunskill. [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1604.00923). In *Proceedings of the 33rd International Conference on Machine Learning*, 2139-2148, 2016.

33. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Off-Policy Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/9541). In *Proceedings of the 9th AAAI Conference on Artificial Intelligence*, 2015.

34. Philip S. Thomas, Georgios Theocharous, and Mohammad Ghavamzadeh. [High Confidence Policy Improvement](https://proceedings.mlr.press/v37/thomas15.html). In *Proceedings of the 32nd International Conference on Machine Learning*, 2380-2388, 2015.

35. Masatoshi Uehara, Jiawei Huang, and Nan Jiang. [Minimax Weight and Q-Function Learning for Off-Policy Evaluation](https://arxiv.org/abs/1910.12809). In *Proceedings of the 37th International Conference on Machine Learning*, 9659--9668, 2020.

36. Masatoshi Uehara, Chengchun Shi, and Nathan Kallus. [A Review of Off-Policy Evaluation in Reinforcement Learning](https://arxiv.org/abs/2212.06355). *arXiv preprint arXiv:2212.06355*, 2022.

37. Mengjiao Yang, Ofir Nachum, Bo Dai, Lihong Li, and Dale Schuurmans. [Off-Policy Evaluation via the Regularized Lagrangian](https://arxiv.org/abs/2007.03438). In *Advances in Neural Information Processing Systems*, 6551--6561, 2020.

38. Christina J. Yuan, Yash Chandak, Stephen Giguere, Philip S. Thomas, and Scott Niekum. [SOPE: Spectrum of Off-Policy Estimators](https://arxiv.org/abs/2111.03936). In *Advances in Neural Information Processing Systems*, 18958--18969, 2022.

39. Shangtong Zhang, Bo Liu, and Shimon Whiteson. [GradientDICE: Rethinking Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2001.11113). In *Proceedings of the 37th International Conference on Machine Learning*, 11194--11203, 2020.

40. Ruiyi Zhang, Bo Dai, Lihong Li, and Dale Schuurmans. [GenDICE: Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2002.09072). In *International Conference on Learning Representations*, 2020.

</details>

<details>
<summary><strong>プロジェクト </strong>(クリックして展開)</summary>

このプロジェクトは，以下の3つのパッケージを参考にしています．
- **Open Bandit Pipeline**  -- 文脈つきバンディットのためのオフ方策評価のパイプライン実装: [[github](https://github.com/st-tech/zr-obp)] [[documentation](https://zr-obp.readthedocs.io/en/latest/)] [[論文](https://arxiv.org/abs/2008.07146)]
- **d3rlpy** -- オフライン強化学習のアルゴリズム実装: [[github](https://github.com/takuseno/d3rlpy)] [[documentation](https://d3rlpy.readthedocs.io/en/v0.91/)] [[論文](https://arxiv.org/abs/2111.03788)]
- **Spinning Up** -- 深層強化学習の学習教材: [[github](https://github.com/openai/spinningup)] [[documentation](https://spinningup.openai.com/en/latest/)]

</details>
