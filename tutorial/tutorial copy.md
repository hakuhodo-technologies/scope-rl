


# オフ方策評価(Off Policy Evaluation)

![スクリーンショット 2023-09-13 12.17.04.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/58151321-c5c1-fd59-7475-07ce3249af4e.png)



**オフ方策評価 (OPE)** とは, 新たな方策の価値を過去の方策から集められたデータのみを利用して評価するというものです. 実際に新たな方策を試すオンライン実験では, 実装コストがかかる, 性能の悪い方策を試してしまうリスクを伴う, 倫理的な観点から難しいなど様々な問題点があります. オフ方策評価は新たな方策をオンライン実験に回さずに評価することができるという点で, 多くの注目が集まっています. オンライン実験をせずにどのように新たな方策の価値を推定するかをオフ方策評価では考えることになります. 

```math
\newcommand{\mE}{\mathbb{E}}
\newcommand{\mR}{\mathbb{R}}
\newcommand{\calD}{\mathcal{D}}
\newcommand{\calA}{\mathcal{A}}
\newcommand{\calI}{\mathcal{I}}
\newcommand{\calS}{\mathcal{S}}
\newcommand{\calT}{\mathcal{T}}
\newcommand{\calP}{\mathcal{P}}
```
# 問題設定
まずは標準的な強化学習の設定であるマルコフ決定過程(MDP）について考えます. 文字の定義については以下の通りです. 

![スクリーンショット 2023-09-13 12.37.55.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/c8fd5678-bd43-1c82-e32d-8633e47f0e7c.png)

<!--
![スクリーンショット 2023-09-13 11.58.42.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/7714b276-fcf2-b9c8-3a47-367bcfd69d00.png)
``` math
\begin{align}
s \in \calS&: 状態\\
a \in \calA &: アクション\\
r \in \mR &: 報酬\\
t = 0, 1, ..., T &:時刻\\

\calT(s'|s, a) &: 状態遷移確率\\
P_r(r|s, a) &: 報酬分布\\
\gamma\in (0,1] &: 割引率\\
d_0  &: 初期の状態分布\\
\end{align}
```

まずは標準的な強化学習での設定マルコフ連鎖(MDP）について考えます. $\calS$は状態空間, $\calA$は連続行動空間または離散行動空間, $\calT:\calS \times \calA \rightarrow \calP(\calS)$は状態遷移確率, $\calT(s' | s,a)$は$s, a$が与えられた時に状態$s'$が観測される確率を表します, 
$P_r: \calS \times \calA \times \mR \rightarrow [0,1]$は即時報酬の確率分布$P_r$が与えられた上で$R: \calS \times \calA \rightarrow \mR $は期待報酬関数を表す. ここで$R(s, a):= \mE_{r \sim P_r(r|s,a)}[r]$は$s$に対して$a$が取られた場合の期待報酬関数になります. 
また$\gamma\in (0,1]$は割引率, $\pi:\calS \rightarrow \calP(\calA)$は方策で$\pi(a|s)$は状態$s$が与えられた時に$a$を撮る確率です, $d_0$は初期の状態分布をあらわしています
オフ方策評価では, ログデータ$\calD$は$n$trajectoriesをもち, それぞれがログ方策$\pi_0$によって
-->

$\pi_0$によって集められた軌跡$\tau$, ログデータ$\mathcal{D}$を以下のように定め, $p_{\pi_0}(\cdot)$を$\pi_0$によって集められた軌跡$\tau$の分布とします. また「軌跡$\tau$の長さ」は時刻の最大である$T$, 「軌跡$\tau$の数」はデータ数として$n$を用います. 
```math
    \tau := \{ (s_t, a_t, s_{t+1}, r_t) \}_{t=0}^{T} \sim  P(d_0) \prod_{t=0}^{T} \pi_0(a_t | s_t) \mathcal{T}(s_{t+1} | s_t, a_t) P_r (r_t | s_t, a_t)

```

```math
\mathcal{D}=\{\tau_i \sim p_{\pi_0}\}_{i=1}^n 
```

オフ方策評価では方策を評価するために, 方策$\pi$の価値を以下のように定義します. 
```math
J(\pi) := \mathbb{E}_{\tau \sim p_{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r_{t} \mid \pi \right ],
```
先ほど説明したように**オフ方策評価では新しい方策$\pi$によるデータを利用せずに, ログデータ$\mathcal{D}$のみを使ってこの$J(\pi)$を推定することになります. ** つまり$J(\pi) \simeq \hat{J}(\pi; D)$となる推定量$\hat{J}(\pi; D)$を考えていくことになります. 推定量の良し悪しは平均二乗誤差(MSE)で評価し
```math
    \begin{aligned}
        \operatorname{MSE}(\hat{J}(\pi  ; \mathcal{D})): & =\mathbb{E}_{\tau  \sim p_{\pi}}\left[(J(\pi)-\hat{J}(\pi ; \mathcal{D}))^2\right] \\
        & =\operatorname{Bias}(\hat{J}(\pi  ; \mathcal{D}))^2+\mathbb{V}_{\tau  \sim p_{\pi}}[\hat{J}(\pi ; \mathcal{D})]
    \end{aligned}
```
MSEはバイアス(bias)とバリアンス(variance)に分けることができるため, この二つの指標をもとに強化学習におけるオフ方策評価の推定量をみていきます. 



# オフ方策評価の推定量
ここからは強化学習の設定におけるオフ方策評価の推定量を紹介し, その性質を理論と簡易実験の両面で解説します. 

- Direct Method 推定量 (DM)
- Trajectory-wise Importance Sampling 推定量 (TIS)
- Per-Decision Importance Sampling 推定量 (PDIS)
- Doubly Robust 推定量 (DR)
- Self-Normalized 推定量 (SN~)
- Marginalized Importance Sampling 推定量(MIS)
- Double Reinforcement Learning 推定量(DRL)
- Spectrum of Off-Policy 推定量(SOPE)

## Direct Method推定量 (DM)
まずは一番基本的な推定量であるDirect Method推定量 (DM)を説明します. DMはFItted Q Evaluation（FQE)などを使って初期の状態価値を推定し, 推定した状態価値を利用した推定量になっています. 

```math
    \large{\hat{J}_{\mathrm{DM}} (\pi; \mathcal{D}) := \mathbb{E}_n [ \mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [\hat{Q}^{\pi}(s_0, a_0)] ] = \mathbb{E}_n [\hat{V}^{\pi}(s_0)]}
```



省略のためここから$\mE_n[f(\tau)] = \frac{1}{n}\sum_{\tau_i \in \calD}f(\tau_i)$の表記を利用します. $\hat{Q}^{\pi}(s_t, a_t) \simeq \mE_{\tau_{t:T}\sim p_{\pi}(\tau_{t:T}|s_t, a_t)}\left[\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}\right]$は推定した行動価値, $\hat{V}^{\pi}(s_t) \simeq \mE_{\tau_{t:T}\sim p_{\pi}(\tau_{t:T}|s_t)}\left[\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}\right]$は推定した状態価値を表します. つまりDMでは方策$\pi$による初期の状態価値を推定することで方策の価値を推定しようとする単純な発想を利用しています. 

DMのバイアス
```math
\begin{align*}
        \operatorname{Bias}[\hat{J}_{\mathrm{DM}}(\pi;D)] & = J(\pi)-  \mathbb{E}_{\tau \sim p_{\pi}}\left[\mathbb{E}_n[\hat{V}^{\pi}(s_0)]\right]\\
& = \mathbb{E}_{\tau \sim p_{\pi}}\left[\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [Q^{\pi}(s_0, a_0)]\right]-  \mathbb{E}_{\tau \sim p_{\pi}}\left[\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [\hat{Q}^{\pi}(s_0, a_0)]\right]\\
& = \mathbb{E}_{\tau \sim p_{\pi}}\left[\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} [Q^{\pi}(s_0, a_0)- \hat{Q}^{\pi}(s_0, a_0)]\right]
\end{align*}
```
DMは他の手法に比べてバリアンスを抑えることができる一方で, 状態価値の推定精度に大きく依存し, 推定誤差$Q^{\pi}(s_0, a_0)- \hat{Q}^{\pi}(s_0, a_0)$が大きい場合は大きなバイアスが発生することになります. 


　
## Trajectory-wise Importance Sampling推定量 (TIS)
DM推定量は大きなバイアスを持つことを紹介しましたが, バイアスを小さくできることで知られているTrajectory-wise Importance Sampling推定量 (TIS)を紹介します. TISは重点サンプリング(importance sampling)という手法を利用しています. 重点サンプリングとは求めたい確率分布の期待値を他の利用できる確率分布からサンプリングされたデータを使って求めようとする手法です. 


```math
\begin{align*}
        & \mathbb{E}_{\tau \sim p_{\pi}}\left[\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &=\sum_{\tau}p_{\pi}(\tau)\sum_{t=0}^{T-1} \gamma^{t}r_t\\
        &=\sum_{\tau}p_{\pi_0}(\tau)\frac{p_{\pi}(\tau)}{p_{\pi_0}(\tau)}\sum_{t=0}^{T-1} \gamma^{t}r_t\\
        &=\mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{p_{\pi}(\tau)}{p_{\pi_0}(\tau)}\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\

\end{align*}
```
今回の例では重点サンプリングを利用することで, 評価方策による分布$p_{\pi}(\tau)$の期待値をログ方策の分布$p_{\pi_0}(\tau)$による期待値に変えることができ, 不偏推定量を設計することが可能になります. この重点サンプリングを利用したTISはこのように定義されます. 


```math
    \large{\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:T-1} r_t \right]}
```


ここで$w_{0:T-1} := \prod_{t=0}^{T-1} (\pi(a_t | s_t) / \pi_0(a_t | s_t))$をtrajectory-wise重要度重みとよび, $\pi(a_t | s_t) / \pi_0(a_t | s_t)$を重要度重みと呼びます. TISではtrajectory wise 重要度重みを利用することで, 共通サポートの仮定$(\prod_{t=0}^{T-1}\pi(a_t \mid s_t) > 0 \rightarrow \prod_{t=0}^{T-1}\pi_0(a_t \mid s_t) > 0)$のもとで不偏性が成り立ちます. 


```math
\mathbb{E}_{\tau \sim p_{\pi}(\tau)}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})] = J(\pi)
```
<details><summary>証明</summary>

```math
\begin{align*}
        &\mathbb{E}_{\tau}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})]\\
        &=\mathbb{E}_{\tau \sim p_{\pi_0}} \left[　\mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t w_{1:T-1} r_t \right] \right]\\
        &=\mathbb{E}_{\tau \sim p_{\pi_0}}\left[\sum_{t=0}^{T-1} \gamma^t w_{1:T-1} r_t \right] \\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{\pi(a_1|s_1)\cdots \pi(a_{T-1}|s_{T-1})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{T-1}|s_{T-1})} \sum_{t=0}^{T-1} \gamma^{t}r_t \right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{p(s_0)\pi(a_1|s_1)P_r(r_1|s_t, a_t)\mathcal{T}(s_{t+1}|s_t, a_t)\cdots \pi(a_{T-1}|s_{T-1})P_r(r_{T-1}|s_{T-1}, a_{T-1})}
        {p(s_0)\pi_0(a_1|s_1)P_r(r_1|s_t, a_t)\mathcal{T}(s_{t+1}|s_t, a_t)\cdots \pi_0(a_{T-1}|s_{T-1})P_r(r_{T-1}|s_{T-1}, a_{T-1})} \sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{p_{\pi}(\tau)}{p_{\pi_0}(\tau)}\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi}}\left[\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &=J(\pi)
\end{align*}
```
</details>



次にバリアンスについてみていきますが, バリアンスを求めるために, TISを再帰的に表現してみます. 
```math
\begin{align*}
J_{\mathrm{TIS}}^{T+1-t} := w_t(w_{t+1:T} r_t + \gamma J_{\mathrm{TIS}}^{T-t})
\end{align*}
```
$J_{\mathrm{TIS}}^{T+1-t}$の$(T+1-t)$は$t$における残りの軌跡の長さを表していて, $J_{\mathrm{TIS}}^0 = 0, J_{\mathrm{TIS}}^T = J_{\mathrm{TIS}}$が成り立ちます. 


<details><summary>式変形</summary>

```math
\begin{align*}
J_{\mathrm{TIS}}^{T+1-t} &= \sum_{t' = t}^{T+1}\gamma^{t' -t}w_{t:T}r_{t'}\\
&=w_{t:T}r_t + \sum_{t' = t+1}^{T+1}\gamma^{t' -t}w_{t:T}r_{t'}\\
&=w_t\left(w_{t+1:T}r_t + \sum_{t' = t+1}^{T+1}\gamma^{t' -t}w_{t+1:T}r_{t'}\right)\\
&=w_t\left(w_{t+1:T}r_t + \gamma\sum_{t' = t+1}^{T+1}\gamma^{t' -(t+1)}w_{t+1:T}r_{t'}\right)\\
&=w_t\left(w_{t+1:T}r_t + \gamma J_{\mathrm{TIS}}^{T+1-(t+1)}\right)\\
&=w_t\left(w_{t+1:T}r_t + \gamma J_{\mathrm{TIS}}^{T-t}\right)\\

\end{align*}
```
</details>


再帰的に表現したことにより求めたTISのバリアンスはこのようになります. 

```math
\mathbb{V}_{t}[\hat{J}_{\mathrm{TIS}}^{T+1-t}(\pi; \mathcal{D})] = \mathbb{V}_t[V(s_t)] + \mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[w_{t+1:T}r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{TIS}}^{T-t}]\right]
```
ここで$w_{t} := \pi(a_{t} | s_{t}) / \pi_0(a_{t} | s_{t})$ ,$\mathbb{E}_t$は下のように定義します. 
```math
\begin{align*}
\mathbb{E}_t:= \mathbb{E}_{s_t, a_t, r_t}[\cdot \mid s_0, a_0, r_0, ..., s_{t-1}, a_{t-1}, r_{t-1}]
\end{align*}
```

<details><summary>証明</summary>

```math

\begin{align*}
    &\mathbb{V}_{t}[\hat{J}_{\mathrm{TIS}}^{T+1-t}(\pi; \mathcal{D})]\\
    &=\mathbb{E}_{t}\left[\left(\hat{J}_{\mathrm{TIS}}^{T+1-t}\right)^2\right]-\Bigl(\mathbb{E}_{t}[V(s_t)]\Bigr)^2 \\
    &=\mathbb{E}_{t}\left[\left(w_t\left(w_{t+1:T}r_t+\gamma \hat{J}_{\mathrm{TIS}}^{T-t} \right)\right)^2\right]-\mathbb{E}_{t}[V(s_t)^2]+\mathbb{V}_t[V(s_t)]\\
    &=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)+w_t\left((w_{t+1:T}r_t+\gamma \hat{J}_{\mathrm{TIS}}^{T-t}-Q(s_t, a_t)\right)\right)^2-V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
    &=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)+w_t\left((w_{t+1:T}r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{TIS}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
    &=\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_t}\left[
    \left(w_tQ(s_t, a_t)+w_t\left((w_{t+1:T}r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{TIS}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right] \biggm\vert s_t, a_t\right]+\mathbb{V}_{t}[V(s_t)]\\
    &=\mathbb{E}_{s_t}\left[\mathbb{E}_{a_t, r_t}\left[
    \left(w_tQ(s_t, a_t)\right)^2 - V(s_t)^2 \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_{t}^2\left((w_{t+1:T}r_t -R(s_t, a_t)\right)^2\right]\right]\\
    &+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_t^2\gamma^2\left(\hat{J}_{\mathrm{TIS}}^{T-t}-\mathbb{E}_{t+1}[V(s_{t+1})]\right)^2\right]\right]+\mathbb{V}_{t}[V(s_t)]\\
    &=\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[w_{t+1:T}r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{TIS}}^{T-t}]\right]+ \mathbb{V}_t[V(s_t)]\\
\end{align*}
```
</details>
重要度重み$w_t$の2乗やバリアンスが含まれており, 特に第３項目$\mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[w_{t+1:T}r_t]\right]$では$w_{t+1:T}$$t+1 \sim T$までの将来の重要度重みの積が含まれており, TISは重要度重みによる非常に大きなバリアンスを持ちます. つまりPDISはバイアスに対しては優れていますが, バリアンスに悩まされます. 

### DM vs TIS (バイアスとバリアンスのトレードオフ）
先ほどまでで理論的にDMはバリアンスは小さいものの, バイアスが大きくなる傾向があり, TISはバイアスが小さいものの, バリアンスが大きくなる傾向があることを説明しました. ここではDMとTISについて簡易実験で性質を紹介します. 軌跡（trajectory)の数$n$を変化させた場合のbias, variance, mseを比較しています. 実験ではスケールを合わせるためbiasはsquared biasを利用します. 
<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/d6444b5c-2ccf-3ee3-2cac-49845a92ca2c.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/c6cb165c-9a6c-5e67-db33-55beae3188ca.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/d728b6e9-6088-cd2b-4d44-bffd2475809c.png" width="300">
</p>
バイアスに関してはDMは大きなバイアスが発生していますが, TISはバイアスがほとんど発生していません. 実験では経験的なバイアスを求めているため, 完全に0にはなりませんが, データ数が増えるごとにTISのバイアスは小さくなり0に近づきます. バリアンスではDMはほとんど発生していない一方で, TISは大きなバリアンスが発生しています. ただしデータ数が増えるごとにバリアンスは小さくなります. バイアスの2乗とバリアンスからなるMSEはデータ数が小さい場合はDMが優れていますが, データ数が増えるごとにTISのバリアンスが小さくなるためTISが優位になります. 

### DM vs TIS (Curse of Horizon）
次は軌跡の長さ$T$を変化させた場合を比較します. y軸は先ほどと違いlogスケールになっている点に注意してください. 

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/2d5add71-745b-2875-895d-188cd0fb5282.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/eab0f3bb-8a14-0615-1204-f2cd1eb9c77d.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/d7b30384-cee6-b63e-38ef-c91dbe4ed41d.png" width="300">
</p>

TISのバリアンスが軌跡が長くなるごとに指数関数的に大きくなってしまっています. つまり軌跡の長さが短い時, TISはDMより優れているものの, 軌跡が長い場合にバリアンスに悩まされることが簡易実験からもわかります. ここまでDMとTISの2つの推定量を見てきましたが, TISを改良した推定量をここから紹介していきます. 


## Per-Decision Importance Sampling推定量 (PDIS)
Per-Decision Importance Sampling推定量 (PDIS)はMDPの性質を利用してTISのバリアンスを抑えようとする推定量です. MDPでは$s_t$は$s_0, ... ,s_{t-1}$や$a_0, ...,a_{t-1}$にのみ依存し, $s_{t+1}, ... ,s_{T}$や$a_{t+1}, ...,a_{T}$には依存しません. つまりPDISでは$r_t$を推定する際に過去$0, ..., t$の重要度重みのみを考慮することを考えます. 
```math
\large{
    \hat{J}_{\mathrm{PDIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \sum_{t=0}^{T-1} \gamma^t w_{0:t} r_t \right]}
```
$w_{0:t} := \prod_{t'=0}^t (\pi(a_{t'} | s_{t'}) / \pi_0(a_{t'} | s_{t'}))$で表され, per-decision重要度重みと呼びます. MDPの性質を利用することで重要度重みの積を軽減しつつ, TISと同様にPDISも共通サポートの仮定のもとで不偏推定量になります. 

```math
    \mathbb{E}_{\tau \sim p_{\pi}}[\hat{J}_{\mathrm{PDIS}} (\pi; \mathcal{D})] = J(\pi)
```
<details><summary>証明</summary>

```math
    \begin{align*}
        \mathbb{E}_{\tau}[\hat{J}_{\mathrm{PDIS}} (\pi; \mathcal{D})]
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\sum_{t=0}^{T-1}\frac{\pi(a_1|s_1)\cdots \pi(a_{t}|s_{t})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{t}|s_{t})} \gamma^{t}r_t \right]\\
        &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim p_{\pi_0}} \left[ \frac{\pi(a_1|s_1)\cdots \pi(a_{t}|s_{t})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{t}|s_{t})} \gamma^{t}r_t  \right] \\
        &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{\pi(a_1|s_1)\cdots \pi(a_{t}|s_{t})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{t}|s_{t})} \gamma^{t}r_t \right]
        \underbrace{\mathbb{E}_{\pi_0(a_1|s_1)\cdots\pi_0(a_t|s_t)}\left[\sum_{a_{t+1}}\cdots\sum_{a_{T-1}}\pi(a_{t+1}|s_{t+1})\cdots\pi(a_{T-1}|s_{T-1})\right]}_{=1} \\
        &= \sum_{t=0}^{T-1} \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{\pi(a_1|s_1)\cdots \pi(a_{t}|s_{t})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{t}|s_{t})} \gamma^{t}r_t \right]
        \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{\pi(a_{t+1}|s_{t+1})\cdots \pi(a_{T-1}|s_{T-1})}
        {\pi_0(a_{t+1}|s_{t+1})\cdots \pi_0(a_{T-1}|s_{T-1})}\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\sum_{t=0}^{T-1}\frac{\pi(a_1|s_1)\cdots \pi(a_{T-1}|s_{T-1})}
        {\pi_0(a_1|s_1)\cdots \pi_0(a_{T-1}|s_{T-1})} \gamma^{t}r_t \right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\frac{p_{\pi}(\tau)}{p_{\pi_0}(\tau)}\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi}}\left[\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &=J(\pi)
    \end{align*}
```
</details>



先ほどと同様にPDISを再帰的に表します. 
```math
\begin{align*}
J_{\mathrm{PDIS}}^{T+1-t} := w_t(r_t + \gamma J_{\mathrm{PDIS}}^{T-t})
\end{align*}
```
$J_{\mathrm{PDIS}}^0 = 0, J_{\mathrm{PDIS}}^T = J_{\mathrm{PDIS}}$が成り立ちます. 

<details><summary>式変形</summary>

```math
\begin{align*}
J_{\mathrm{PDIS}}^{T+1-t} &= \sum_{t' = t}^{T+1}\gamma^{t' -t}w_{t:t'}r_{t'}\\
&=w_tr_t + \sum_{t' = t+1}^{T+1}\gamma^{t' -t}w_{t:t'}r_{t'}\\
&=w_t\left(r_t + \sum_{t' = t+1}^{T+1}\gamma^{t' -t}w_{t+1:t'}r_{t'}\right)\\
&=w_t\left(r_t + \gamma\sum_{t' = t+1}^{T+1}\gamma^{t' -(t+1)}w_{t+1:t'}r_{t'}\right)\\
&=w_t(r_t + \gamma J_{\mathrm{PDIS}}^{T+1-(t+1)})\\
&=w_t(r_t + \gamma J_{\mathrm{PDIS}}^{T-t})\\
\end{align*}
```
</details>


PDISのバリアンスを計算すると以下のようになります. 

```math
    \mathbb{V}_{t}[\hat{J}_{\mathrm{PDIS}}^{T+1-t}(\pi; \mathcal{D})] = \mathbb{V}_t[V(s_t)] +\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{PDIS}}^{T-t}]\right]
```

<details><summary>証明</summary>

```math
\begin{align*}
        &\mathbb{V}_{t}[\hat{J}_{\mathrm{PDIS}}^{T+1-t}(\pi; \mathcal{D})]\\
        &=\mathbb{E}_{t}\left[\left(\hat{J}_{\mathrm{PDIS}}^{T+1-t}\right)^2\right]-\Bigl(\mathbb{E}_{t}[V(s_t)]\Bigr)^2 \\
        &=\mathbb{E}_{t}\left[\left(w_t\left(r_t+\gamma \hat{J}_{\mathrm{PDIS}}^{T-t} \right)\right)^2\right]-\mathbb{E}_{t}[V(s_t)^2]+\mathbb{V}_t[V(s_t)]\\
        &=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)+w_t\left(r_t+\gamma \hat{J}_{\mathrm{PDIS}}^{T-t}-Q(s_t, a_t)\right)\right)^2-V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{PDIS}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_t}\left[
        \left(w_tQ(s_t, a_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{PDIS}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right] \biggm\vert s_t, a_t\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t}\left[\mathbb{E}_{a_t, r_t}\left[
        \left(w_tQ(s_t, a_t)\right)^2 - V(s_t)^2 \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_{t}^2\left(r_t -R(s_t, a_t)\right)^2\right]\right]\\
        &+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_t^2\gamma^2\left(\hat{J}_{\mathrm{PDIS}}^{T-t}-\mathbb{E}_{t+1}[V(s_{t+1})]\right)^2\right]\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{PDIS}}^{T-t}]\right]+ \mathbb{V}_t[V(s_t)]\\
\end{align*}
```
</details>

TISのバリアンス(再掲）

```math
\mathbb{V}_{t}[\hat{J}_{\mathrm{TIS}}^{T+1-t}(\pi; \mathcal{D})] = \mathbb{V}_t[V(s_t)] + \mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[w_{t+1:T}r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{TIS}}^{T-t}]\right]
```

先ほど求めたTISのバリアンスと比較してみると第3項のみ異なっており, PDISはTISでの$r_t$の係数$w_{t+1:T}$が除かれていることがわかります. PDISでtrajectory-wise重要度重みからper-decision重要度重みに変えたことによってバリアンスを抑えることができています. 

<!--
先ほど求めたTISのバリアンスと比較してみると第3項のみ異なっており, TISでの$\mathbb{E}_{s_t,a_t}[w^2_t \mathbb{V}_{r_{t+1}}[w_{t+1:T}r_t]]$ だった部分から $w_{t+1:T}$が除かれていることがわかります. PDISでtrajectory-wise重要度重みからper-decision重要度重みに変えたことによってバリアンスを抑えることができています. 
-->

### TIS vs PDIS 
TISとPDISの比較について簡易実験で確認します. 

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/9fbbafd6-1b97-1ce1-e7dd-2f17b327caa2.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/5aaf4f2a-8866-ba47-2d29-5e134ebc3d69.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/2b8d9a97-45cf-fa76-b5f4-d56119ef6fb7.png" width="300">
</p>

PDISはバイアスをTISと同等に保ちつつも(バリアンスに比べればバイアスの差は微小）, バリアンスを抑えることができています. したがってMSEを見ても分かるようにTISよりもPDISの方が優れた推定量であると言えます. しかし, PDISであったとしても軌跡の長さ$T$が大きい場合は依然としてバリアンスが大きくなってしまいます. 

## Doubly Robust推定量 (DR)

Doubly Robust推定量 (DR)ではバリアンスに強いDMとバイアスに強いPDISを組み合わせた推定量になっています. 再帰的に表現されたPDISに$\hat{Q}$をベースラインとして組み込むことを考えます.


```math
\begin{align*}
J_{\mathrm{DR}}^{T+1-t} := \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)] + w_t(r_t + \gamma J_{\mathrm{DR}}^{T-t} - \hat{Q}^{\pi}(s_t, a_t))
\end{align*}
```
TISやPDISと同様に式変形をすることで以下のDRが提案されます. 

```math
\large{\hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})
:= \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)])\right]}
```
DRはPDISの性質を引き継ぎ, 共通サポートの仮定のもとで不偏性を持ちます

```math
\mathbb{E}_{\tau \sim p_{\pi}}[\hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})] = J(\pi)
```
<details><summary>証明</summary>

```math
    \begin{align*}
        &\mathbb{E}_{\tau}[\hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t \left (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)]\right)\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})]  - \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t}\hat{Q}(s_t, a_t) \right] + \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t-1} \mathbb{E}_{a \sim \pi_0(a | s_t)}\left[\frac{\pi(a \mid s_t)}{\pi_0(a \mid s_t)}\hat{Q}(s_t, a)\right]\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})]  - \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t}\hat{Q}(s_t, a_t) \right] + \mathbb{E}_{\tau \sim { (s_{t'}, s_{t'+1}, r_{t'}) \}_{t'=0}^{T-1}}} \prod_{t' = 0}^{T-1}\mathbb{E}_{a \sim \pi_0(\cdot | s_{t'})}\left [\sum_{t=0}^{T-1} \gamma^t w_{0:t-1} \mathbb{E}_{a \sim \pi_0(a | s_t)}\left[\frac{\pi(a \mid s_t)}{\pi_0(a \mid s_t)}\hat{Q}(s_t, a)\right]\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})]  - \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t}\hat{Q}(s_t, a_t) \right] + \mathbb{E}_{\tau \sim { (s_{t'}, s_{t'+1}, r_{t'}) \}_{t'=0}^{T-1}}} \prod_{t' = 0}^{T-1}\mathbb{E}_{a \sim \pi_0(\cdot | s_{t'})}\left [\sum_{t=0}^{T-1} \gamma^t w_{0:t-1} \frac{\pi(a_t \mid s_t)}{\pi_0(a_t \mid s_t)}\hat{Q}(s_t, a_t)\right]\\
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}[\hat{J}_{\mathrm{TIS}} (\pi; \mathcal{D})]  - \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t}\hat{Q}(s_t, a_t) \right] + \mathbb{E}_{\tau \sim p_{\pi_0}} \left[\sum_{t=0}^{T-1} \gamma^t w_{0:t}\hat{Q}(s_t, a_t)) \right] \\
        &= J(\pi)
    \end{align*}
```
</details>

DRのバリアンス

```math
\mathbb{V}_{t}[\hat{J}_{\mathrm{DR}}^{T+1-t}(\pi; \mathcal{D})] = \mathbb{V}_t[V(s_t)]+\mathbb{E}_{s_t}\left[\mathbb{V}_{a_t, r_t}\left[w_t(\hat{Q}(s_t, a_t)-Q(s_t, a_t)) \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[{w_t}^2\mathbb{V}_{r_{t+1}}[r_t]\right] + \mathbb{E}_{s_t, a_t}\left[\gamma^2{w_t}^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{DR}}^{T-t}]\right] 
```



<details><summary>証明</summary>

```math
    \begin{align*}
        &\mathbb{V}_{t}[\hat{J}_{\mathrm{DR}}^{T+1-t}(\pi; \mathcal{D})]\\
        &=\mathbb{E}_{t}\left[\left(\hat{J}_{\mathrm{DR}}^{T+1-t}\right)^2\right]-\Bigl(\mathbb{E}_{t}[V(s_t)]\Bigr)^2 \\
        &=\mathbb{E}_{t}\left[\left(\hat{V}(s_t)+w_t\left(r_t+\gamma \hat{J}_{\mathrm{DR}}^{T-t} - \hat{Q}(s_t, a_t)\right)\right)^2\right]-\mathbb{E}_{t}[V(s_t)^2]+\mathbb{V}_t[V(s_t)]\\
        &=\mathbb{E}_{t}\left[\left(w_tQ(s_t, a_t)-w_t\hat{Q}(s_t, a_t)+\hat{V}(s_t)+w_t\left(r_t+\gamma \hat{J}_{\mathrm{DR}}^{T-t}-Q(s_t, a_t)\right)\right)^2-V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{t}\left[\left(w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{DR}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_t}\left[
        \left(w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)+w_t\left(r_t-R(s_t, a_t)\right)+w_t\gamma \left(\hat{J}_{\mathrm{DR}}^{T-t} -\mathbb{E}_{t+1}[V(s_{t+1})]\right)\right)^2 -V(s_t)^2\right] \biggm\vert s_t, a_t\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t}\left[\mathbb{E}_{a_t, r_t}\left[
        \left(-w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t)\right)^2 - V(s_t)^2 \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_{t}^2\left(r_t -R(s_t, a_t)\right)^2\right]\right]\\
        &+\mathbb{E}_{s_t, a_t}\left[\mathbb{E}_{r_{t+1}}\left[w_t^2\gamma^2\left(\hat{J}_{\mathrm{DR}}^{T-t}-\mathbb{E}_{t+1}[V(s_{t+1})]\right)^2\right]\right]+\mathbb{V}_{t}[V(s_t)]\\
        &=\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ -w_t(Q(s_t, a_t)-\hat{Q}(s_t, a_t))+\hat{V}(s_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{DR}}^{T-t}]\right]+ \mathbb{V}_t[V(s_t)]\\
        &=\mathbb{E}_{s_t}\left[\mathbb{V}_{a_t, r_t}\left[w_t(\hat{Q}(s_t, a_t)-Q(s_t, a_t)) \mid s_t\right]\right]+\mathbb{E}_{s_t, a_t}\left[{w_t}^2\mathbb{V}_{r_{t+1}}[r_t]\right] + \mathbb{E}_{s_t, a_t}\left[\gamma^2{w_t}^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{DR}}^{T-t}]\right] + \mathbb{V}_t[V(s_t)] 
    \end{align*}
```
</details>

PDISのバリアンス(再掲）
```math
    \mathbb{V}_{t}[\hat{J}_{\mathrm{PDIS}}^{T+1-t}(\pi; \mathcal{D})] = \mathbb{V}_t[V(s_t)] +\mathbb{E}_{s_t} \left[ \mathbb{V}_{a_t, r_t} \left [ w_tQ(s_t, a_t) \mid s_t \right] \right ] + \mathbb{E}_{s_t,a_t} \left[w_t^2\mathbb{V}_{r_{t+1}}[r_t]\right]+\mathbb{E}_{s_t, a_t}\left[ w_t^2 \gamma^2\mathbb{V}_{r_{t+1}}[\hat{J}_{\mathrm{PDIS}}^{T-t}]\right]
```

PDISのバリアンスと比較してみると第2項が異なっていることがわかります. DRでは$w_t$の係数に$(\hat{Q}(s_t, a_t)-Q(s_t, a_t))$が組み込まれていることにより$\hat{Q}$の精度が良いほどバリアンスの減少させることができます. 具体的には$\hat{Q}(\cdot)$が$0<\hat{Q}(\cdot)<2Q(\cdot)$を満たせばDRはPDISよりもバリアンスの小さい推定量になります. 

### PDIS vs DR
簡易実験でも軌跡の長さを変えた場合のPDISとTISを比較してみます. バリアンスがPDISに比べDRでは$\hat{Q}(\cdot)$を入れたことで, 少し小さくなっていることが実験でもわかります. 
<!--
<p align="center">
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/1e4c03c0-559e-4a44-09fa-16bce26fce97.png" width="300"> 
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/8275c7c8-fc36-3cde-7654-8bca9a48e6ef.png" width="300">
<img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/43afeb24-dd10-9580-1197-bea7a0e57e15.png" width="300">
</p>
-->

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/11d09b20-e507-95a9-17ba-fd5e3943c881.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/ead36c6e-cc7e-62c6-0565-a5873d30c455.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/d305817f-8ca9-1e95-ce9f-ab5abc0b3214.png" width="300">
</p>


## Self-Normalized推定量

Self-Normalized推定量はバリアンスを減らすために, 重要度重みのスケールを小さくしようとするモチベーションから生まれています. 具体的には重要度重み$w_{\ast}$を以下に置き換えたものを使います

```math
    \tilde{w}_{\ast} := w_{\ast} / \mathbb{E}_{n}[w_{\ast}]
```
ここで$\tilde{w}_{\ast} $はself-normalized重要度重みと呼ばれます. Self-Normalized推定量は今まで重要度重みを利用するすべての推定量で利用することができます. 

### TIS vs SNTIS
ここではSelf-Normalized推定量の中でSelf-Normalized TIS(SNTIS)とTISを比較します. 

<!--
<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/c9b46cba-abc1-b055-3d9f-f285acb07666.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/0a1debc3-a191-891b-3233-103d8044bd5e.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/93eadcea-0f87-59c8-4983-776fe735a401.png" width="300">
</p>
-->

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/75035cc3-b08f-613f-9c63-81f1e04406f0.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/20fb808f-9f0c-22b1-9561-00f6eb6414f5.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/b67997de-4878-1b01-6d7e-911df84ab0be.png" width="300">
</p>
バイアスはTISと同程度に保ちつつ, バリアンスを大幅に減らすことに成功しています. 結果としてMSEも改善されています. ここまでTIS→PDIS→DR→Self-Normalizedと見てきましたが, バリアンスの発生原因になっていた重要度重みをどのように変形するかを考えてきました. ただしここまでの推定量では重要度重みが軌跡の長さに依存してしまっているため, 軌跡の長さが長くなった場合の根本的な解決にはなっていないという問題点があります. 



## Marginalized Importance Sampling推定量
今までの推定量では軌跡の長さが長い場合に対応できないという問題点が存在しました. これに対して重要度重み自体を変え, state marginalまたはstate-action marginal重要度重みを利用したMarginalized Importance Sampling推定量が提案されています. 
<!--
それぞれ重要度重みは

```math
\begin{align*}
    w_{s, a}(s, a) &:= d^{\pi}(s, a) / d^{\pi_0}(s, a) \\
    w_s(s) &:= d^{\pi}(s) / d^{\pi_0}(s)
\end{align*}
```


```math
\begin{align*}
    w(s_t, a_t) &= w_{s, a}(s_t, a_t) \\
    w(s_t, a_t) &= w_{s}(s_t) w_{t}(s_t, a_t)
\end{align*}
```
$d_t^{\pi}(s_t, a_t) := p_{\pi}(s_t, a_t)$

$d_t^{\pi}(s_t) := p_{\pi}(s_t)$
-->


```math
\begin{align*}
    w_{s, a}(s_t, a_t) &:= d_t^{\pi}(s_t, a_t) / d_t^{\pi_0}(s_t, a_t) \\
    w_s(s_t) &:= d_{t}^{\pi}(s_t)\pi(a_t|s_t)/ d_t^{\pi_0}(s_t)\pi_0(a_t|s_t)
\end{align*}
```
$d_t^{\pi}(s_t, a_t)$は方策$\pi$での$s_t, a_t$となる確率, $d_t^{\pi}(s_t)$は方策$\pi$での$s_t$となる確率を表します. 
上のように重要度重みを定めた時, 推定量は以下のように定義されます. 


```math
\large{
    \hat{J}_{\mathrm{SAMIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \sum_{t=0}^{T-1} \gamma^t w_{s, a}(s_t, a_t) r_t \right]}
```

```math
\large{
    \hat{J}_{\mathrm{SMIS}} (\pi; \mathcal{D}) := \mathbb{E}_{n} \left[ \sum_{t=0}^{T-1} \gamma^t w_{s}(s_t) r_t \right]}
```

共通サポートの仮定のもとでSAMISとSMISは不偏性を満たします. 

```math
\mathbb{E}_{\tau}[\hat{J}_{\mathrm{SAMIS}} (\pi; \mathcal{D})]= J(\pi)
```


<details><summary>証明</summary>

$d^{\pi}(s, a) := \left(\sum_{t=1}^L \gamma^{t-1} d_t^\pi(s, a)\right) /\left(\sum_{t=1}^L \gamma^{t-1}\right)$を利用します
```math

\begin{align*}
        \mathbb{E}_{\tau}[\hat{J}_{\mathrm{SAMIS}} (\pi; \mathcal{D})]
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\sum_{t=0}^{T-1}\frac{d_t^{\pi}(s_t, a_t)}
        {d_t^{\pi_0}(s_t, a_t)} \gamma^{t}r_t \right]\\
        &= \sum_{s, a}\sum_{t=0}^{T-1}d_t^{\pi_0}(s_t, a_t)\frac{d^{\pi}(s, a)}
        {d^{\pi_0}(s, a)} \gamma^{t}R(s, a) \\
        &=\left( \sum_{t=0}^{T-1}\gamma^{t}\right)\sum_{s, a}d^{\pi_0}(s, a)\frac{d^{\pi}(s, a)}
        {d^{\pi_0}(s, a)} R(s, a) \\
        &=\left( \sum_{t=0}^{T-1}\gamma^{t}\right)\sum_{s, a}d^{\pi}(s, a) R(s, a) \\
        &= \sum_{s, a}\sum_{t=0}^{T-1}d_t^{\pi}(s_t, a_t)\gamma^{t}R(s, a) \\
        &= \mathbb{E}_{\tau \sim p_{\pi}}\left[\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &=J(\pi)
\end{align*}

```
</details>


```math
\mathbb{E}_{\tau}[\hat{J}_{\mathrm{SMIS}} (\pi; \mathcal{D})]= J(\pi)
```

<details><summary>証明</summary>

$d^{\pi}(s) := \left(\sum_{t=1}^L \gamma^{t-1} d_t^\pi(s)\right) /\left(\sum_{t=1}^L \gamma^{t-1}\right)$を利用します
```math

\begin{align*}
        \mathbb{E}_{\tau}[\hat{J}_{\mathrm{SMIS}} (\pi; \mathcal{D})]
        &= \mathbb{E}_{\tau \sim p_{\pi_0}}\left[\sum_{t=0}^{T-1}\frac{d_t^{\pi}(s_t)\pi(a_t | s_t)}
        {d_t^{\pi_0}(s_t)\pi_0(a_t | s_t)} \gamma^{t}r_t \right]\\
        &= \sum_{s, a}\sum_{t=0}^{T-1}d_t^{\pi_0}(s_t, a_t)\pi_0(a_t | s_t)\frac{d^{\pi}(s)\pi(a_t | s_t)}
        {d^{\pi_0}(s)\pi_0(a_t | s_t)} \gamma^{t}R(s, a) \\
        &= \sum_{s, a}\sum_{t=0}^{T-1}d_t^{\pi_0}(s_t, a_t)\pi(a_t | s_t)\frac{d^{\pi}(s)}
        {d^{\pi_0}(s)} \gamma^{t}R(s, a) \\
        &=\left( \sum_{t=0}^{T-1}\gamma^{t}\right)\sum_{s, a}d^{\pi_0}(s)\frac{d^{\pi}(s)}
        {d^{\pi_0}(s)} \pi(a | s)R(s, a) \\
        &=\left( \sum_{t=0}^{T-1}\gamma^{t}\right)\sum_{s, a}d^{\pi}(s)  \pi(a | s)R(s, a) \\
        &= \sum_{s, a}\sum_{t=0}^{T-1}d_t^{\pi}(s_t)\pi(a_t|s_t)\gamma^{t}R(s, a) \\
        &= \mathbb{E}_{\tau \sim p_{\pi}}\left[\sum_{t=0}^{T-1} \gamma^{t}r_t\right]\\
        &=J(\pi)
\end{align*}

```
</details>


<!--
この推定量は異なるtrajectoryにおいて, 同じまたは似ている状態を推移する場合に使いやすくなります. 例えば状態推移が$\cdots \rightarrow s_1 \rightarrow s_2 \rightarrow s_1 \rightarrow s_2 \rightarrow \cdots$ または$ \cdots \rightarrow s_{\ast} \rightarrow s_1 \rightarrow s_{\ast} \rightarrow \cdots$
-->

Marginal推定量はmarginal重要度重み$w_{s,a}(s,a)), w_s(s)$が既知の場合, 不偏推定量となりますが, 多くのケースで真の重要度重みを利用することができません. したがってmarginal重要度重みを推定し, 推定結果をを利用することになります. この場合重要度重みの推定誤差によるバイアスが発生することになります. 

### PDIS vs SAMIS
ここではState-Action Marginal推定量のSAMISとPDISを比較していきます. 
<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/0a24236c-15a0-4575-180a-e6fc52a93a1f.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/45f6993e-c92c-7d0e-e251-5855ec3fc30b.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/6d909711-8f2b-3e4a-8652-a36de758f88b.png" width="300">
</p>
SAMISはstate-action marginal重要度重みの推定によるバイアスが発生していますが, ここまでどの推定量でも問題になっていた軌跡の長さが長い場合のバリアンスを大幅に減らすことができているため, 重要度重みを使いつつバリアンスを減らすことができる推定量になっていることがわかります. 

最後に今までの推定量を組み合わせた発展的な推定量として2つほど紹介します. 
## Double Reinforcement Learning 推定量(DRL)
まずper-decision重要度重みを利用したstandard DR推定量はこのように定義されていました. 
standard DR
```math
\hat{J}_{\mathrm{DR}} (\pi; \mathcal{D})
:= \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t (w_{0:t} (r_t - \hat{Q}(s_t, a_t)) + w_{0:t-1} \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_t, a)]) \right]
```
またmarginal重要度重みを利用したmarginal DRはState-Action MarginalDR推定量としてこのように定義されています. 


marginal DR 
```math
\begin{align*}
\hat{J}_{\mathrm{SAM-DR}} (\pi; \mathcal{D})
&:= \mathbb{E}_{n} [\mathbb{E}_{a_0 \sim \pi(a_0 | s_0)} \hat{Q}(s_0, a_0)] \\
& \quad \quad + \mathbb{E}_{n} \left[\sum_{t=0}^{T-1} \gamma^t w_{s, a}(s_t, a_t) (r_t + \gamma \mathbb{E}_{a \sim \pi(a | s_t)}[\hat{Q}(s_{t+1}, a)] - \hat{Q}(s_t, a_t)) \right]
\end{align*}
```
ここで自然な発想としてmarginal重要度重みをstandard DRに組み込んだ新たな推定量としてDoubleReinforcementLearnig推定量 （DRL)が以下のように提案されています. 


```math
\begin{align*}
    \large{\hat{J}_{\mathrm{DRL}} (\pi; \mathcal{D})}
    &\large{ := \frac{1}{n} \sum_{k=1}^K \sum_{i=1}^{n_k} \sum_{t=0}^{T-1} (w_s^j(s_{i,t}, a_{i, t}) (r_{i, t} - Q^j(s_{i, t}, a_{i, t})) }\\
    & \large{\quad \quad + w_s^j(s_{i, t-1}, a_{i, t-1}) \mathbb{E}_{a \sim \pi(a | s_t)}[Q^j(s_{i, t}, a)] )}
\end{align*}
```
ここでは$Q$から発生する潜在的なバイアスを緩和するために"cross-fitting"という手法を利用します. 具体的にはfoldの数$K$を選択し, ログデータ$\mathcal{D}$を$n_k$で分割した$j$番目のデータを$\mathcal{D}_j$とします. cross-fittingでは$w^j$や$Q^j$をデータの部分集合$ \mathcal{D}\setminus \mathcal{D}_j$で学習します. これにより過学習などをおこすことなく学習することができます. 

### DR vs DRL
DRとDRLを軌跡の長さを変化させ, 簡易実験によって比較します. 

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/48504d48-7b2a-45ac-3f2a-72101dcec72f.png" width="300"> 
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/8e2a4755-2dc4-f089-aef5-666afd17575b.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/b6fe607d-4f93-fef6-e86b-9f62585a87ca.png" width="300">
</p>

DRLは軌跡の長さが長くなった場合でもmarginal重要度重みを使うことでper-decesion重要度重みを利用しているDRよりもバリアンスを抑えることができています. DRLはQ関数の推定がうまくいけば行くほど, バリアンスを減らすことができます. 

## Spectrum of Off-Policy 推定量(SOPE)
marginal重要度重みは, per-decesion重要度重みのバリアンスを効果的に緩和しますが, marginal重要度重みの推定誤差は推定にバイアスをもたらす可能性があることを紹介しました. これに対して, バイアスとバリアンスのトレードオフをより柔軟に制御するために, Spectrum of Off-Policy 推定量(SOPE)は以下のような重要度重みを使用します. 
```math
\begin{align*}
    w(s_t, a_t) &= 
    \begin{cases}
        \prod_{t'=0}^{k-1} w_t(s_{t'}, a_{t'}) & \mathrm{if} \, t < k \\
        w_{s, a}(s_{t-k}, a_{t-k}) \prod_{t'=t-k+1}^{t} w_t(s_{t'}, a_{t'}) & \mathrm{otherwise}
    \end{cases} \\
    w(s_t, a_t) &= 
    \begin{cases}
        \prod_{t'=0}^{k-1} w_t(s_{t'}, a_{t'}) & \mathrm{if} \, t < k \\
        w_{s}(s_{t-k}) \prod_{t'=t-k}^{t} w_t(s_{t'}, a_{t'}) & \mathrm{otherwise}
    \end{cases}
\end{align*}
```
$t<k$の場合はPDISと同じper-decesion重要度重みを利用し, $t \geq k$の場合はMISと同じmarginal重要度重みを利用しています. つまりPDISとMISを組み合わせた推定量としてSOPEは提案されています. 


### PDIS vs SAMIS vs SOPE
簡易実験ではSOPEをSAMISとPDISに対して比較しています. 横軸は$k$でSOPEがどこまでpdisと同じ重要度重みを使うかを表しているためn_step_pdisとしています. $k$を変えることでMarginal推定量とPDIS推定量のバランスをコントロールすることができます. 

<p align="center">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/ea8bd9d9-c9f4-9d6d-beb0-f3dd3fe8aec0.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/6ecccbce-a3ac-df43-50c3-e21e06ecd0ca.png" width="300">
    <img src="https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/3535628/4d603583-cdbb-c448-c36d-4763d923656b.png" width="300"> 
</p>

図から分かるようにSOPEは$k=0$のときSAMISと一致し, $k=T$(軌跡の長さ)のときPDISと一致します. $k$を大きくするとバイアスを抑えられ, $k$を小さくするとバリアンスを抑えることができます. SOPEは$k$を調節することでバイアスとバリアンスのトレードオフを考慮でき, SAMIS,PDISよりMSEが小さくなるように設計することが可能です. 


