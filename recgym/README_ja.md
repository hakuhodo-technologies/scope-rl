# RECGym: 推薦システムの強化学習環境
<details>
<summary><strong>目次 </strong>(クリックして展開)</summary>

- [RECGym: 推薦システムの強化学習環境](#RECGym-推薦システムの強化学習環境)
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

*RECGym* は推薦設定での強化学習シミュレーション環境です. このシミュレータは [OpenAI Gym](https://gym.openai.com) および [Gymnasium](https://gymnasium.farama.org/) のインターフェイスに従っています．RECGymは，`UserModel`(`user_preference_dynamics`や`reward_function`を実装) といった内部モジュールをカスタマイズできるように設計されています．

RECGymは [scope-rl](../) リポジトリの下で公開されており，オフライン強化学習の実装を試す環境として容易に使えます．

### 標準設定

推薦システムにおける累積報酬を最大化の問題を(部分観測)マルコフ決定過程((PO)MDP)として定式化します．
- `状態`: 
   - ユーザーの持つ特徴ベクトルで，エージェントが提示する行動に応じて時間と共に変化する.
   - 真の状態が観測できない場合，エージェントは状態の代わりにノイズがのった観測を用いる．
- `行動`: どのアイテムをユーザに提示するかを表す．
- `報酬`: ユーザーの興味の大きさを表す．0/1の二値または連続量．

### 実装

RECGymでは，推薦環境の標準的な環境を提供しています．
- `"RECEnv-v0"`: 標準的な推薦環境.

RECGym は以下の環境で構成されています．
- [RECEnv](./envs/rec.py#L14): 大枠となる環境．カスタマイズ可能な環境設定を引数に持つ．

RECGymは以下のモジュールについて設定可能です．
- [UserModel](./envs/simulator/function.py#L13): 推薦システムのユーザーモデルを定義するクラス．

上記のモジュールは [abstract class](./envs/simulator/base.py) に従ってカスタマイズすることができます．

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

標準環境とカスタマイズされた環境の使用例を提供します．
なお，オンライン/オフライン強化学習およびオフ方策評価の実装例は，[SCOPE-RLのREADME](../README.md)で紹介されています．

### 標準的な推薦環境

標準的なRECEnvは，[OpenAI Gym](https://gym.openai.com) や [Gymnasium](https://gymnasium.farama.org/)のインターフェースに従い `gym.make()` から利用可能です．

```Python
# recgymとgymをインポートする
import recgym
import gym

# (1) 標準的な環境
env = gym.make('RECEnv-v0')
```

基本的なインタラクションは，以下の4行のコードのみで実行できます．

```Python
obs, info = env.reset()
while not done:
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)
```

ランダム方策で行うインタラクションを可視化してみましょう．

```Python
# 他のライブラリからインポートする
from scope_rl.policy import OnlineHead
from d3rlpy.algos import DiscreteRandomPolicy

# ランダムエージェントを定義する
agent = OnlineHead(
    DiscreteRandomPolicy(),
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
ax1.set_xlabel('timestep')
ax1.set_ylabel('reward')
ax1.legend(loc='upper left')
plt.show()
```
<div align="center"><img src="./images/basic_interaction.png" width="60%"/></div>
<figcaption>
<p align="center">
  一エピソードにおける報酬の変遷
</p>
</figcaption>

今回の例では [SCOPE-RL](../README.md) と [d3rlpy](https://github.com/takuseno/d3rlpy) を利用していますが，BasicGymは[OpenAI Gym](https://gym.openai.com) と [Gymnasium](https://gymnasium.farama.org/)のインターフェースに対応している他のライブラリとも互換性があります．

### カスタマイズしたRECEnv

次に，環境のカスタマイズの方法を説明します．

<details>
<summary>環境設定のリスト: (クリックして展開)</summary>

- `step_per_episode`: 一つのエピソードでの意思決定の数
- `n_items`: 推薦システムでのアイテムの数
- `n_users`: 推薦システムでのユーザーの数
- `item_feature_dim`: アイテム特徴量の次元
- `user_feature_dim`: ユーザー特徴量の次元
- `item_feature_vector`: それぞれのアイテムの特徴量(ベクトル)
- `user_feature_vector`: それぞれのユーザーの特徴量(ベクトル)
- `reward_type`: 報酬のタイプ
- `reward_std`: 報酬のノイズの大きさ (reward_typeが"continuous"の場合のみ)
- `obs_std`: 状態観測のノイズの大きさ
- `StateTransitionFunction`: 状態遷移関数
- `UserModel`: ユーザーモデル (ユーザーの嗜好の推移と報酬関数を定義)
- `random_state` : ランダムシード

</details>

```Python
from recgym import RECEnv
env = RECEnv(
    step_per_episode=10,
    n_items=100,
    n_users=100,
    item_feature_dim=5,
    user_feature_dim=5,
    reward_type="continuous",  # "binary"
    reward_std=0.3,
    obs_std=0.3,
    random_state=12345,
)
```

また，以下のように独自の `UserModel`を定義・使用できます．

#### ユーザーモデルの例
```Python
# reccgymモジュールをインポートする
from recgym import BaseUserModel
from recgym.types import Action
# その他必要なものをインポートする
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class CustomizedUserModel(BaseUserModel):
    user_feature_dim: int
    item_feature_dim: int
    reward_type: str = "continuous"  # "binary"
    reward_std: float = 0.0
    random_state: Optional[int] = None

    def __post_init__(self):
        self.random_ = check_random_state(self.random_state)
        self.coef = self.random_.normal(size=(self.user_feature_dim, self.item_feature_dim))

    def user_preference_dynamics(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        coefficient = state.T @ self.coef @ item_feature_vector[action]
        state = state + alpha * coefficient * item_feature_vector[action]
        state = state / np.linalg.norm(state, ord=2)
        return state

    def reward_function(
        self,
        state: np.ndarray,
        action: Action,
        item_feature_vector: np.ndarray,
    ) -> float:
        logit = state.T @ self.coef @ item_feature_vector[action]
        reward = (
            logit if self.reward_type == "continuous" else sigmoid(logit)
        )

        if self.reward_type == "discrete":
            reward = self.random_.binominal(1, p=reward)

        return reward
```

より多くの例は [quickstart_ja/rec/rec_synthetic_customize_env_ja.ipynb](./examples/quickstart_ja/rec/rec_synthetic_customize_env_ja.ipynb)を参照してください．\
環境の統計量の可視化は，[quickstart_ja/rec/rec_synthetic_data_collection_ja.ipynb](./examples/quickstart_ja/rec/rec_synthetic_data_collection_ja.ipynb)で確認できます．

## 引用

ソフトウェアを使用する場合は，以下の論文の引用をお願いします．

Haruka Kiyohara, Ren Kishimoto, Kosuke Kawakami, Ken Kobayashi, Kazuhide Nakata, Yuta Saito.<br>
**SCOPE-RL: A Python Library for Offline Reinforcement Learning and Off-Policy Evaluation**<br>

Bibtex:
```
@article{kiyohara2023scope,
  author = {Kiyohara, Haruka and Kishimoto, Ren and Kawakami, Kosuke and Kobayashi, Ken and Nataka, Kazuhide and Saito, Yuta},
  title = {SCOPE-RL: A Python Library for Offline Reinforcement Learning and Off-Policy Evaluation},
  journal={arXiv preprint arXiv:2311.18206},
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

3. Sarah Dean and Jamie Morgenstern. [Preference Dynamics Under Personalized Recommendations](https://arxiv.org/abs/2205.13026). In *Proceedings of the 23rd ACM Conference on Economics and Computation*, 4503-9150, 2022.

</details>

<details>
<summary><strong>プロジェクト </strong>(クリックして展開)</summary>

このプロジェクトは，以下の4つのプロジェクトを参考にしています.
- **RecoGym**  -- 推薦システムのための強化学習環境: [[github](https://github.com/criteo-research/reco-gym)] [[論文](https://arxiv.org/abs/1808.00720)]
- **RecSim** -- 推薦システムのためのカスタマイズ可能な強化学習環境: [[github](https://github.com/google-research/recsim)] [[論文](https://arxiv.org/abs/1909.04847)]
- **AuctionGym** -- 広告入札のための強化学習環境: [[github](https://github.com/amzn/auction-gym)] [[論文](https://www.amazon.science/publications/learning-to-bid-with-auctiongym)]
- **FinRL** -- 金融・投資のための強化学習環境: [[github](https://github.com/AI4Finance-Foundation/FinRL)] [[論文](https://arxiv.org/abs/2011.09607)]

</details>

