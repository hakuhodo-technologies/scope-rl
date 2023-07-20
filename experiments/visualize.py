from pathlib import Path
import pickle
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import check_random_state
from cycler import cycler

import hydra
from omegaconf import DictConfig
import matplotlib.gridspec as gridspec


color_dict = {
    "red": "#E24A33",
    "blue": "#348ABD",
    "purple": "#988ED5",
    "gray": "#777777",
    "green": "#8EBA42",
    "yellow": "#FBC15E",
    "pink": "#FFB5B8",
}
cd = color_dict

plt.style.use("ggplot")
colors = [cd["red"], cd["blue"], cd["purple"], cd["gray"], cd["yellow"]]
plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
markers = ["o", "v", "^", "s", "p", "P", "*", "h", "X", "D", "d"]


def normalized_conventional_metrics(
    env_name: str,
    conventional_metrics_dict: Dict[str, Any],
    behavior_name: str,
):
    n_dataset = 10
    estimators = ["dm", "snpdis", "sndr", "sam_snis", "sam_sndr"]
    ESTIMATORS = ["DM", "PDIS", "DR", "MIS", "MDR"]

    regret = {j: {k: 0 for k in ESTIMATORS} for j in range(n_dataset)}
    mse = {j: {k: 0 for k in ESTIMATORS} for j in range(n_dataset)}
    rank_cc = {j: {k: 0 for k in ESTIMATORS} for j in range(n_dataset)}
    best_true_policy_value = np.max(
        conventional_metrics_dict[behavior_name][0][estimators[0]]["true_policy_value"]
    )
    mean_true_policy_value = np.mean(
        conventional_metrics_dict[behavior_name][0][estimators[0]]["true_policy_value"]
        ** 2
    )

    for i in range(n_dataset):
        for j, estimator in enumerate(estimators):
            regret[i][ESTIMATORS[j]] = conventional_metrics_dict[behavior_name][i][
                estimator
            ]["regret"][0] / abs(best_true_policy_value)
            mse[i][ESTIMATORS[j]] = (
                conventional_metrics_dict[behavior_name][i][estimator][
                    "mean_squared_error"
                ]
                / mean_true_policy_value
            )
            rank_cc[i][ESTIMATORS[j]] = conventional_metrics_dict[behavior_name][i][
                estimator
            ]["rank_correlation"][0]

    regret_df = pd.DataFrame(regret)
    regret_df["mean"] = regret_df.mean(axis=1)
    regret_df["std"] = regret_df.std(axis=1)

    rank_cc_df = pd.DataFrame(rank_cc)
    rank_cc_df["mean"] = rank_cc_df.mean(axis=1)
    rank_cc_df["std"] = rank_cc_df.std(axis=1)

    mse_df = pd.DataFrame(mse)
    mse_df["mean"] = mse_df.mean(axis=1)
    mse_df["std"] = mse_df.std(axis=1)

    conventional_total_df = pd.concat(
        [
            mse_df[["mean", "std"]],
            rank_cc_df[["mean", "std"]],
            regret_df[["mean", "std"]],
        ],
        axis=1,
        keys=["rMSE", "Rank_cc", "rRegret"],
    ).round(3)

    return conventional_total_df, regret_df, rank_cc_df, mse_df


def visualize_sharpe_ratio(
    env_name: str,
    topk_metrics_dict: Dict[str, Any],
    sharpe_log_scale: bool = False,
    behavior_on_policy: float = 0.0,
    max_topk: int = 10,
    alpha: float = 0.05,
    plot_ymin: float = -0.2,
    plot_ymax: float = 5.0,
    random_state: Optional[int] = None,
    log_dir: Optional[str] = None,
):
    estimators = ["dm", "snpdis", "sndr", "sam_snis", "sam_sndr"]
    ESTIMATORS = ["DM", "PDIS", "DR", "MIS", "MDR"]
    random_ = check_random_state(random_state)
    path_ = Path(log_dir + "results/sharpe_ratio")
    path_.mkdir(exist_ok=True, parents=True)
    save_path = Path(path_ / f"sharpe_ratio_{env_name}.png")

    lower_sharpe = np.zeros(max_topk)
    upper_sharpe = np.zeros(max_topk)
    lower_std = np.zeros(max_topk)
    upper_std = np.zeros(max_topk)
    lower_best = np.zeros(max_topk)
    upper_best = np.zeros(max_topk)
    n_bootstrap_samples = 100
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_colors = len(colors)

    fig = plt.figure(constrained_layout=True, figsize=(12.0, 3.5))
    gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3.8, 6.2])

    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
    ax00 = fig.add_subplot(gs00[0])

    for i, estimator in enumerate(estimators):
        metric_dict = topk_metrics_dict[estimator]
        sharpe_ratio = metric_dict["sharpe_ratio"][1:]
        sharpe_ratio = np.nan_to_num(sharpe_ratio, posinf=1e2)
        sharpe_ratio = np.clip(sharpe_ratio, 0.0, 1e2)

        ax00.plot(
            np.arange(2, max_topk + 1),
            sharpe_ratio.mean(axis=1),
            color=colors[i],
            marker=markers[i],
            label=ESTIMATORS[i],
        )

        ax00.plot(
            np.arange(2, max_topk + 1),
            np.zeros((max_topk - 1,)),
            color="black",
            linewidth=0.5,
        )

        for topk in range(1, max_topk):
            samples = metric_dict["sharpe_ratio"][topk]
            samples = np.nan_to_num(samples, posinf=1e2)
            samples = np.clip(samples, 0.0, 1e2)
            boot_samples = [
                np.mean(random_.choice(samples, size=samples.shape[0]))
                for i in range(n_bootstrap_samples)
            ]
            lower_sharpe[topk] = np.percentile(boot_samples, 100 * (alpha / 2))
            upper_sharpe[topk] = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))

        ax00.fill_between(
            np.arange(2, max_topk + 1),
            lower_sharpe[1:],
            upper_sharpe[1:],
            color=colors[i % n_colors],
            alpha=0.3,
        )

        if sharpe_log_scale:
            ax00.set_yscale("log")
        ax00.set_title(f"SharpeRatio@k")
        ax00.set_xlabel("# of policies deployed", fontsize=14)
        ax00.set_ylabel(f"SharpeRatio", fontsize=14)
        ax00.set_ylim((plot_ymin, plot_ymax))
        # ax00.legend(loc=sharpe_legend)

    gs01 = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs0[1], height_ratios=[1, 4]
    )
    gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs01[1], wspace=0.3)
    ax01 = fig.add_subplot(gs02[0])
    ax02 = fig.add_subplot(gs02[1])

    for i, estimator in enumerate(estimators):
        metric_dict = topk_metrics_dict[estimator]

        ax01.plot(
            np.arange(2, max_topk + 1),
            np.maximum(
                metric_dict["best"][1:].mean(axis=1) - (behavior_on_policy),
                np.zeros((max_topk - 1,)),
            ),
            color=colors[i],
            marker=markers[i],
            label=ESTIMATORS[i],
        )

        ax02.plot(
            np.arange(2, max_topk + 1),
            metric_dict["std"][1:].mean(axis=1),
            color=colors[i],
            marker=markers[i],
            label=ESTIMATORS[i],
        )

        for topk in range(1, max_topk):
            samples_best = np.maximum(
                metric_dict["best"][topk] - (behavior_on_policy), np.zeros((max_topk,))
            )
            boot_samples_best = [
                np.mean(random_.choice(samples_best, size=samples_best.shape[0]))
                for i in range(n_bootstrap_samples)
            ]
            lower_best[topk] = np.percentile(boot_samples_best, 100 * (alpha / 2))
            upper_best[topk] = np.percentile(boot_samples_best, 100 * (1.0 - alpha / 2))

            samples_std = metric_dict["std"][topk]
            boot_samples_std = [
                np.mean(random_.choice(samples_std, size=samples_std.shape[0]))
                for i in range(n_bootstrap_samples)
            ]
            lower_std[topk] = np.percentile(boot_samples_std, 100 * (alpha / 2))
            upper_std[topk] = np.percentile(boot_samples_std, 100 * (1.0 - alpha / 2))

        ax01.fill_between(
            np.arange(2, max_topk + 1),
            lower_best[1:],
            upper_best[1:],
            color=colors[i % n_colors],
            alpha=0.3,
        )

        ax02.fill_between(
            np.arange(2, max_topk + 1),
            lower_std[1:],
            upper_std[1:],
            color=colors[i % n_colors],
            alpha=0.3,
        )

    ax01.set_title(f"numerator (best@$k$ - $J(\pi_b)$)")
    ax01.set_xlabel("# of policies deployed", fontsize=14)
    ax01.set_ylabel(f"best@$k$ - $J(\pi_b)$", fontsize=14)
    # ax01.legend(loc=best_legend)

    ax02.set_title(f"denominator (std@$k$)")
    ax02.set_xlabel("# of policies deployed", fontsize=14)
    ax02.set_ylabel(f"std@$k$", fontsize=14)
    # ax02.legend(loc=std_legend)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def visualize_topk_statistics(
    env_name: str,
    topk_metrics_dict: Dict[str, Any],
    behavior_on_policy: float = 0.0,
    max_topk: int = 10,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
    log_dir: Optional[str] = None,
):
    estimators = ["dm", "snpdis", "sndr", "sam_snis", "sam_sndr"]
    metrics = ["best", "worst", "mean", "std"]

    random_ = check_random_state(random_state)
    path_ = Path(log_dir + "results/topk_statistics")
    path_.mkdir(exist_ok=True, parents=True)
    save_path = Path(path_ / f"topk_metrics_{env_name}.png")

    lower = np.zeros(max_topk)
    upper = np.zeros(max_topk)
    n_bootstrap_samples = 100
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n_colors = len(colors)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(5 * 3, 3),
    )

    metric_dict = topk_metrics_dict[estimators[0]]
    max_val = metric_dict["best"].mean(axis=1).max()
    min_val = metric_dict["worst"].mean(axis=1).min()

    yaxis_min_val = min(min_val, behavior_on_policy)
    yaxis_max_val = max(max_val, behavior_on_policy)
    margin = (yaxis_max_val - yaxis_min_val) * 0.05

    for i, estimator in enumerate(estimators):
        for j, metric in enumerate(metrics):
            metric_dict = topk_metrics_dict[estimator]

            for topk in range(max_topk):
                samples = metric_dict[metric][topk]
                boot_samples = [
                    np.mean(random_.choice(samples, size=samples.shape[0]))
                    for i in range(n_bootstrap_samples)
                ]
                lower[topk] = np.percentile(boot_samples, 100 * (alpha / 2))
                upper[topk] = np.percentile(boot_samples, 100 * (1.0 - alpha / 2))

            if metric == "std":
                axes[j].plot(
                    np.arange(2, max_topk + 1),
                    metric_dict[metric][1:].mean(axis=1),
                    color=colors[i],
                    marker=markers[i],
                )
                axes[j].fill_between(
                    np.arange(2, max_topk + 1),
                    lower[1:],
                    upper[1:],
                    color=colors[i % n_colors],
                    alpha=0.3,
                )

            else:
                axes[j].plot(
                    np.arange(1, max_topk + 1),
                    metric_dict[metric].mean(axis=1),
                    color=colors[i],
                    marker=markers[i],
                )
                axes[j].fill_between(
                    np.arange(1, max_topk + 1),
                    lower,
                    upper,
                    color=colors[i % n_colors],
                    alpha=0.3,
                )
                axes[j].plot(
                    np.arange(1, max_topk + 1),
                    np.full(max_topk, max_val),
                    color="black",
                    linewidth=0.5,
                )
                axes[j].plot(
                    np.arange(1, max_topk + 1),
                    np.full(max_topk, min_val),
                    color="black",
                    linewidth=0.5,
                )
                axes[j].plot(
                    np.arange(1, max_topk + 1),
                    np.full(max_topk, behavior_on_policy),
                    color="#A60628",
                    label="safety threshold",
                )

            axes[j].set_xlabel("# of policies deployed", fontsize=14)
            axes[j].set_title(f"{metric}", fontsize=16)

            if metric == "std":
                axes[j].set_ylabel(f"{metric} of policy values", fontsize=14)
            else:
                axes[j].set_ylabel(f"{metric} policy value", fontsize=14)

            if metric != "std":
                axes[j].set_ylim(yaxis_min_val - margin, yaxis_max_val + margin)

            # axes[j].legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def visualize_benchmark_results(
    env_name: str,
    normalized_conventional_total_df: pd.DataFrame,
    topk_metrics_dict: Dict[str, Any],
    ymax_mse: Optional[float] = None,
    ymin_rankcorr: float = -1.0,
    ymax_rankcorr: float = 1.0,
    ymax_sharpe: float = 5.0,
    random_state: Optional[int] = None,
    log_dir: Optional[str] = None,
):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(3 * 5, 3),
    )

    estimators = ["dm", "snpdis", "dr", "sam_snis", "sam_sndr"]
    ESTIMATORS = {
        "dm": "DM",
        "snpdis": "PDIS",
        "dr": "DR",
        "sndr": "DR",
        "sam_snis": "MIS",
        "sam_sndr": "MDR",
    }

    path_ = Path(log_dir + "results/benchmark")
    path_.mkdir(exist_ok=True, parents=True)
    save_path = Path(path_ / f"benchmark_{env_name}.png")

    METRICS = {"rMSE": "nMSE", "Rank_cc": "RankCorr", "rRegret": "nRegret@1"}
    for i, metric in enumerate(["rMSE", "Rank_cc", "rRegret"]):
        for j, estimator in enumerate(estimators):
            axes[i].bar(
                np.arange(j, j + 1),
                normalized_conventional_total_df[metric]["mean"].to_numpy()[j],
                yerr=normalized_conventional_total_df[metric]["std"].to_numpy()[j],
                color=colors[j],
                label=ESTIMATORS[estimator],
                alpha=0.85,
            )

        axes[i].set_title(METRICS[metric], fontsize=16)
        axes[i].set_xticks(np.arange(5), [""] * 5)
        axes[i].tick_params(axis="y", labelsize=12)

    for j, estimator in enumerate(estimators):
        if estimator == "dr":
            estimator = "sndr"
        axes[3].bar(
            np.arange(j, j + 1),
            topk_metrics_dict[estimator]["sharpe_ratio"][4].mean(),
            yerr=topk_metrics_dict[estimator]["sharpe_ratio"][4].std(),
            color=colors[j],
            label=ESTIMATORS[estimator],
            alpha=0.85,
        )
        axes[4].bar(
            np.arange(j, j + 1),
            topk_metrics_dict[estimator]["sharpe_ratio"][7].mean(),
            yerr=topk_metrics_dict[estimator]["sharpe_ratio"][7].std(),
            color=colors[j],
            label=ESTIMATORS[estimator],
            alpha=0.85,
        )
        axes[3].set_title("SharpeRatio@4", fontsize=16)
        axes[3].set_xticks(np.arange(5), [""] * 5)
        axes[3].tick_params(axis="y", labelsize=12)
        axes[4].set_title("SharpeRatio@7", fontsize=16)
        axes[4].set_xticks(np.arange(5), [""] * 5)
        axes[4].tick_params(axis="y", labelsize=12)

    axes[0].set_ylim(0.0, ymax_mse)
    axes[1].set_ylim(ymin_rankcorr, ymax_rankcorr)
    axes[2].set_ylim(0.0, None)
    axes[3].set_ylim(0.0, ymax_sharpe)
    axes[4].set_ylim(0.0, ymax_sharpe)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")


def process(
    env_name: str,
    behavior_sigma: float,
    behavior_tau: float,
    random_state: int,
    log_dir: str,
):
    path_ = Path(log_dir + f"/results")
    path_.mkdir(exist_ok=True, parents=True)

    if env_name in ["Hopper", "Reacher", "InvertedPendulum", "Swimmer"]:
        with open(
            path_ / f"topk/topk_metrics_dict_{env_name}_sac_gauss_{behavior_sigma}.pkl",
            "rb",
        ) as f:
            topk_metrics_dict = pickle.load(f)

        with open(
            path_
            / f"behavior/behavior_on_policy_{env_name}_sac_gauss_{behavior_sigma}.pkl",
            "rb",
        ) as f:
            on_policy = pickle.load(f)

        with open(
            path_
            / f"conventional/conventional_metrics_dict_{env_name}_sac_gauss_{behavior_sigma}.pkl",
            "rb",
        ) as f:
            conventional_metrics_dict = pickle.load(f)

        (
            normalized_conventional_total_df,
            regret_df,
            rank_cc_df,
            mse_df,
        ) = normalized_conventional_metrics(
            env_name, conventional_metrics_dict, f"sac_gauss_{behavior_sigma}"
        )

    else:
        with open(
            path_
            / f"topk/topk_metrics_dict_{env_name}_ddqn_softmax_{behavior_tau}.pkl",
            "rb",
        ) as f:
            topk_metrics_dict = pickle.load(f)

        with open(
            path_
            / f"behavior/behavior_on_policy_{env_name}_ddqn_softmax_{behavior_tau}.pkl",
            "rb",
        ) as f:
            on_policy = pickle.load(f)

        with open(
            path_
            / f"conventional/conventional_metrics_dict_{env_name}_ddqn_softmax_{behavior_tau}.pkl",
            "rb",
        ) as f:
            conventional_metrics_dict = pickle.load(f)

        (
            normalized_conventional_total_df,
            regret_df,
            rank_cc_df,
            mse_df,
        ) = normalized_conventional_metrics(
            env_name, conventional_metrics_dict, f"ddqn_softmax_{behavior_tau}"
        )

    if env_name in ["Hopper", "Swimmer", "CartPole"]:
        visualize_sharpe_ratio(
            env_name,
            topk_metrics_dict,
            behavior_on_policy=on_policy,
            random_state=random_state,
            log_dir=log_dir,
            plot_ymax=2.5,
        )
    else:
        visualize_sharpe_ratio(
            env_name,
            topk_metrics_dict,
            behavior_on_policy=on_policy,
            random_state=random_state,
            log_dir=log_dir,
        )

    visualize_topk_statistics(
        env_name,
        topk_metrics_dict,
        behavior_on_policy=on_policy,
        random_state=random_state,
        log_dir=log_dir,
    )

    visualize_benchmark_results(
        env_name,
        normalized_conventional_total_df,
        topk_metrics_dict,
        random_state=random_state,
        log_dir=log_dir,
        ymin_rankcorr=-0.65,
        ymax_rankcorr=1.1,
        ymax_sharpe=3.6,
    )


def assert_configuration(cfg: DictConfig):
    env_name = cfg.setting.env_name
    assert env_name in [
        "HalfCheetah",
        "Hopper",
        "InvertedPendulum",
        "Reacher",
        "Swimmer",
        "CartPole",
        "MountainCar",
        "Acrobot",
    ]

    behavior_sigma = cfg.setting.behavior_sigma
    assert isinstance(behavior_sigma, float) and behavior_sigma > 0.0

    behavior_tau = cfg.setting.behavior_tau
    assert isinstance(behavior_tau, float) and behavior_tau != 0.0

    random_state = cfg.setting.base_random_state
    assert isinstance(random_state, int) and random_state > 0


@hydra.main(config_path="conf/", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    assert_configuration(cfg)
    print(f"The current working directory is {Path().cwd()}")
    print(f"The original working directory is {hydra.utils.get_original_cwd()}")
    print()
    conf = {
        "env_name": cfg.setting.env_name,
        "behavior_sigma": cfg.setting.behavior_sigma,
        "behavior_tau": cfg.setting.behavior_tau,
        "random_state": cfg.setting.base_random_state,
        "log_dir": "logs/",
    }
    process(**conf)


if __name__ == "__main__":
    main()
