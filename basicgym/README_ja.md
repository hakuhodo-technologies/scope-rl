# BasicGym: 簡易的な強化学習環境
<details>
<summary><strong>目次 </strong>(クリックして展開)</summary>

- [BasicGym: 簡易的な強化学習環境](#basicgym-簡易的な強化学習環境)
- [概要](#概要)
- [インストール](#インストール)
- [用法](#用法)
- [引用](#引用)
- [貢献](#貢献)
- [ライセンス](#ライセンス)
- [プロジェクトチーム](#プロジェクトチーム)
- [連絡先](#連絡先)
- [参考文献](#参考文献)

</details>

## 概要

*BasicGym* は強化学習の簡易なシミュレーションのための環境です．このシミュレーターは特に強化学習アルゴリズム用に設計されており，[OpenAI Gym](https://gym.openai.com) および [Gymnasium](https://gymnasium.farama.org/) のインターフェイスに従っています．BasicGymは，`StateTransitionFunction` や `RewardFunction`を含む環境モジュールをカスタマイズできるように設計されています．

BasicGymは [scope-rl](../) リポジトリの下で公開されており，オフライン強化学習の実装を試す環境として容易に使えます．

### 標準設定

(部分観測)マルコフ決定過程((PO)MDP)として定式化します．
- `状態`: 
    状態観測 (POMDPの場合は観測ノイズが発生) ．
- `行動`: 
    強化学習エージェント (方策) により選択された行動．
- `報酬`: 
    状態と行動に応じて観測される報酬．

### 実装

BasicGymでは，離散的および連続的な行動空間の両方において標準的な環境を提供しています．
- `"BasicEnv-continuous-v0"`: 連続行動空間に対する標準的な環境.
- `"BasicEnv-discrete-v0"`: 離散行動空間に対する標準的な環境.

BasicGym は以下の環境で構成されています．
- [BasicEnv](./envs/basic.py#L18): 大枠となる環境．カスタマイズ可能な環境設定を引数に持つ．

BasicGymでは以下のモジュールをカスタマイズできます．
- [StateTransitionFunction](./envs/simulator/function.py#L14): 状態遷移関数を定義するクラス．
- [RewardFunction](./envs/simulator/function.py#L101): 期待報酬関数を定義するクラス．

なお，カスタマイズはBasicGymで実装されている [abstract class](./envs/simulator/base.py) に従って行うことができます．


## インストール

BasicGymは，Pythonの`pip`を使用して [scope-rl](../) の一部としてインストールすることができます．
```
pip install scope-rl
```

また，コードからインストールすることもできます．
```bash
git clone https://github.com/hakuhodo-technologies/scope-rl
cd scope-rl
python setup.py install
```

## 用法

標準実装の環境を用いる例とカスタマイズした環境を用いる例を紹介します．
なお，オンライン/オフライン強化学習およびオフ方策評価の実装例は，[SCOPE-RLのREADME](../README.md)で紹介されています．

### 標準的な BasicEnv

標準実装のBasicEnvは，[OpenAI Gym](https://gym.openai.com) や [Gymnasium](https://gymnasium.farama.org/)のインターフェースに従い， `gym.make()` から利用可能です．

```Python
# basicgymとgymをインポートする
import basicgym
import gym

# (1) 標準的な環境
env = gym.make('BasicEnv-continuous-v0')
```

基本的なインタラクションは，以下の4行のコードのみで実行できます．

```Python
obs, info = env.reset()
while not done:
    action = agent.sample_action_online(obs)
    obs, reward, done, truncated, info = env.step(action)
```

ランダム方策で行うインタラクションを可視化してみましょう．

```Python
# 他のライブラリからインポートする
from scope_rl.policy import OnlineHead
from d3rlpy.algos import RandomPolicy as ContinuousRandomPolicy

# ランダムエージェントを定義する
agent = OnlineHead(
    ContinuousRandomPolicy(
        action_scaler=MinMaxActionScaler(
            minimum=0.1,  # 方策が取り得る最小値
            maximum=10,  # 方策が取り得る最大値
        )
    ),
    name="random",
)
agent.build_with_env(env)

# (2) 基本的なインタラクション
obs, info = env.reset()
done = False
# ログ
reward_list = []

while not done:
    action = agent.sample_action_online(obs)
    obs, reward, done, truncated, info = env.step(action)
    # ログ
    reward_list.append(reward)


# 結果を可視化する
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(reward_list[:-1], label='reward', color='tab:orange')
ax1.set_xlabel('time_step')
ax1.set_ylabel('reward')
ax1.legend(loc='upper left')
plt.show()
```
<div align="center"><img src="./images/basic_interaction.png" width="60%"/></div>
<figcaption>
<p align="center">
   一エピソード中に観察された報酬
</p>
</figcaption>

今回の例では [SCOPE-RL](../README.md) と [d3rlpy](https://github.com/takuseno/d3rlpy) を利用していますが，BasicGymは[OpenAI Gym](https://gym.openai.com) と [Gymnasium](https://gymnasium.farama.org/)のインターフェースに対応している他のライブラリとも互換性があります．

### カスタマイズしたBasicEnv

次に，環境のカスタマイズの方法を説明します．

<details>
<summary>環境設定のリスト: (クリックして展開)</summary>

- `step_per_episode`: 一つのエピソードでの意思決定の数
- `state_dim`: 状態の次元
- `action_type`: 強化学習エージェントの行動のタイプ
- `n_actions`: 離散行動空間での行動の数
- `action_dim`: 行動の次元
- `action_context`: それぞれの行動を表す特徴ベクトル (action_typeが"discrete"の場合のみ)
- `reward_type`: 報酬のタイプ
- `reward_std`: 報酬のノイズの大きさ (reward_typeが"continuous"の場合のみ)
- `obs_std`: 状態観測のノイズの大きさ
- `StateTransitionFunction`: 状態遷移関数
- `RewardFunction`: 報酬関数
- `random_state` : ランダムシード

</details>

```Python
from basicgym import BasicEnv
env = BasicEnv(
    state_dim=10,
    action_type="continuous",  # "discrete"
    action_dim=5,
    reward_type="continuous",  # "binary"
    reward_std=0.3,
    obs_std=0.3,
    step_per_episode=10,
    random_state=12345,
)
```

また，以下のように独自の `StateTransitionFunction` と `RewardFunction` を定義・使用できます．

#### 状態遷移関数のカスタマイズの例
```Python
# basicgymモジュールをインポートする
from basicgym import BaseStateTransitionFunction
# その他必要なものをインポートする
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CustomizedStateTransitionFunction(BaseStateTransitionFunction):
    state_dim: int
    action_dim: int
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.state_dim))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, self.action_dim))

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> np.ndarray:
        state = self.state_coef @ state / self.state_dim +  self.action_coef @ action / self.action_dim
        state = state / np.linalg.norm(state, ord=2)
        return state

```

#### 報酬関数の例
```Python
# basicgymモジュールをインポートする
from basicgym import BaseRewardFunction
# その他必要なものをインポートする
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CustomizedRewardFunction(BaseRewardFunction):
    state_dim: int
    action_dim: int
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.state_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.state_dim, ))
        self.action_coef = self.random_.normal(loc=0.0, scale=1.0, size=(self.action_dim, ))

    def mean_reward_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        reward = self.state_coef.T @ state / self.state_dim + self.action_coef.T @ action / self.action_dim
        return reward
```

より多くの例は[quickstart_ja/basic_synthetic_customize_env_ja.ipynb](./examples/quickstart_ja/basic_synthetic_customize_env_ja.ipynb)を参照してください．

## 引用

ソフトウェアを使用する場合は，以下の論文をお願いします．

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection**<br>
[link]() (a preprint coming soon..)

Bibtex:
```
@article{kiyohara2023towards,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {SCOPE-RL: A Python Library for Offline Reinforcement Learning, Off-Policy Evaluation, and Policy Selection},
  journal={arXiv preprint arXiv:23xx.xxxxx},
  year = {2023},
}
```

## 貢献

SCOPE-RLへの貢献も歓迎しています！
プロジェクトへの貢献方法については， [CONTRIBUTING.md](./CONTRIBUTING.md)を参照してください．

## ライセンス

このプロジェクトはApache 2.0ライセンスのもとでライセンスされています - 詳細については[LICENSE](LICENSE)ファイルをご覧ください．

## プロジェクトチーム

- [清原 明加 (Haruka Kiyohara)](https://sites.google.com/view/harukakiyohara) (コーネル大学，**Main Contributor**)
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

1. Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. [OpenAI Gym](https://arxiv.org/abs/1606.01540). *arXiv preprint arXiv:1606.01540*, 2016.

2. Takuma Seno and Michita Imai. [d3rlpy: An Offline Deep Reinforcement Library](https://arxiv.org/abs/2111.03788), *arXiv preprint arXiv:2111.03788*, 2021.


</details>

<details>
<summary><strong>プロジェクト </strong>(クリックして展開)</summary>

このプロジェクトは，以下のパッケージを参考にしています．
- **Open Bandit Pipeline**  -- 文脈つきバンディットにおけるオフ方策評価のパイプライン実装: [[github](https://github.com/st-tech/zr-obp)] [[documentation](https://zr-obp.readthedocs.io/en/latest/)] [[論文](https://arxiv.org/abs/2008.07146)]

</details>


