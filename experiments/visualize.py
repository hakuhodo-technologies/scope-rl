import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.utils import check_random_state

def sharpe_ratio(
    env_name, 
    behavior_name,
    # topk_metrics_dict, 
    sharpe_log_scale=False, 
    best_log_scale=False, 
    std_log_scale=False, 
    on_policy_value=-38, 
    sharpe_legend="lower right",
    best_legend="lower right",
    std_legend="lower right",
    max_topk=10, 
    alpha=0.05
    ):
  
  with open(f"logs/results/raw/topk_metrics_dict_{env_name}_{behavior_name}_10.pkl", "rb") as f:
    topk_metrics_dict = pickle.load(f)

  estimator_list = ['dm', 'snpdis', 'sndr', 'sam_snis']
  estimator_name = ['DM', 'PDIS', 'DR', 'MIS']
  random_ = check_random_state(12345)

  lower_sharpe = np.zeros(max_topk)
  upper_sharpe = np.zeros(max_topk)
  lower_std = np.zeros(max_topk)
  upper_std = np.zeros(max_topk)
  lower_best = np.zeros(max_topk)
  upper_best = np.zeros(max_topk)
  n_bootstrap_samples = 1000
  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
  n_colors = len(colors)

  fig = plt.figure(constrained_layout=True, figsize=(12.0, 3.5))
  gs0 = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3.8,6.2])

  gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0])
  ax00 = fig.add_subplot(gs00[0])

  for i, estimator in enumerate(estimator_list):
    metric_dict = topk_metrics_dict[estimator]

    ax00.plot(
        np.arange(2, max_topk + 1),
        metric_dict['sharpe_ratio'][1:].mean(axis=1),
        color=colors[i],
        marker=markers[i],
        label=estimator_name[i],
    )

    ax00.plot(
        np.arange(2, max_topk + 1),
        np.zeros((max_topk - 1, )),
        color="black",
        linewidth=0.5,
    )

    for topk in range(1, max_topk):
      samples = metric_dict['sharpe_ratio'][topk]
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
          alpha=0.03,
      )
    if sharpe_log_scale:
        ax00.set_yscale('log')
    ax00.set_title(f"Sharpe ratio")
    ax00.set_xlabel("\# of policies deployed", fontsize=14)
    ax00.set_ylabel(f"Sharpe ratio", fontsize=14)
    ax00.legend(loc=sharpe_legend)

  gs01 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs0[1], height_ratios=[1,4])
  gs02 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs01[1], wspace=0.2)
  ax01 = fig.add_subplot(gs02[0])
  ax02 = fig.add_subplot(gs02[1])

  for i, estimator in enumerate(estimator_list):
    metric_dict = topk_metrics_dict[estimator]

    ax01.plot(
        np.arange(2, max_topk + 1),
        metric_dict["best"][1:].mean(axis=1) - (on_policy_value),
        color=colors[i],
        marker=markers[i],
        label=estimator_name[i],
    )

    ax02.plot(
        np.arange(2, max_topk + 1),
        metric_dict["std"][1:].mean(axis=1),
        color=colors[i],
        marker=markers[i],
        label=estimator_name[i],
    )
  
    for topk in range(1, max_topk):
      samples_best = metric_dict['best'][topk] - (on_policy_value)
      boot_samples_best = [
          np.mean(random_.choice(samples_best, size=samples_best.shape[0]))
          for i in range(n_bootstrap_samples)
      ]
      lower_best[topk] = np.percentile(boot_samples_best, 100 * (alpha / 2))
      upper_best[topk] = np.percentile(boot_samples_best, 100 * (1.0 - alpha / 2))

      ax01.fill_between(
          np.arange(2, max_topk + 1),
          lower_best[1:],
          upper_best[1:],
          color=colors[i % n_colors],
          alpha=0.03,
      )

      samples_std = metric_dict['std'][topk]
      boot_samples_std = [
          np.mean(random_.choice(samples_std, size=samples_std.shape[0]))
          for i in range(n_bootstrap_samples)
      ]
      lower_std[topk] = np.percentile(boot_samples_std, 100 * (alpha / 2))
      upper_std[topk] = np.percentile(boot_samples_std, 100 * (1.0 - alpha / 2))

      ax02.fill_between(
          np.arange(2, max_topk + 1),
          lower_std[1:],
          upper_std[1:],
          color=colors[i % n_colors],
          alpha=0.03,
      )

  if best_log_scale:
      ax01.set_yscale('log')
  ax01.set_title(f"numerator (best@$k$ - $J(\pi_b)$)")
  ax01.set_xlabel("\# of policies deployed", fontsize=14)
  ax01.set_ylabel(f"best@$k$ - $J(\pi_b)$", fontsize=14)
  ax01.legend(loc=best_legend)

  if std_log_scale:
      ax02.set_yscale('log')
  ax02.set_title(f"denominator (std@$k$)")
  ax02.set_xlabel("\# of policies deployed", fontsize=14)
  ax02.set_ylabel(f"std@$k$", fontsize=14)
  # ax02.set_ylim(10, 30)
  ax02.legend(loc=std_legend)

  fig.tight_layout()
  fig.savefig(f"sharpe_ratio_{env_name}.png", dpi=300, bbox_inches="tight")