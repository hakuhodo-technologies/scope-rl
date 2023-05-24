import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
from sklearn.utils import check_random_state


def conventional_metrics(env_name, conventional_metrics_dict, behavior_name):
  n_dataset = 10
  estimator_list = ['dm', 'snpdis', 'sndr', 'sam_snis']

  with open(f"logs/results/raw/conventional_metrics_dict_{env_name}_{behavior_name}_10.pkl", "rb") as f:
    conventional_metrics_dict = pickle.load(f)
  
  regret = {j: {k: 0 for k in estimator_list} for j in range(n_dataset)}
  mse = {j: {k: 0 for k in estimator_list}for j in range(n_dataset)}
  rank_cc = {j: {k: 0 for k in estimator_list}for j in range(n_dataset)}

  for i in range(n_dataset):
    for estimator in estimator_list:
      regret[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['regret'][0]
      mse[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['mean_squared_error']
      rank_cc[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['rank_correlation'][0]

  regret_df = pd.DataFrame(regret)
  regret_df['mean'] = regret_df.mean(axis=1)
  regret_df['std'] = regret_df.std(axis=1)

  rank_cc_df = pd.DataFrame(rank_cc)
  rank_cc_df['mean'] = rank_cc_df.mean(axis=1)
  rank_cc_df['std'] = rank_cc_df.std(axis=1)

  mse_df = pd.DataFrame(mse)
  mse_df['mean'] = mse_df.mean(axis=1)
  mse_df['std'] = mse_df.std(axis=1)

  return regret_df, rank_cc_df, mse_df 


def relative_conventional_metrics(env_name, conventional_metrics_dict, behavior_name):
  n_dataset = 10
  estimator_list = ['dm', 'snpdis', 'sndr', 'sam_snis']
  
  with open(f"logs/results/raw/conventional_metrics_dict_{env_name}_{behavior_name}_10.pkl", "rb") as f:
    conventional_metrics_dict = pickle.load(f)

  regret = {j: {k: 0 for k in estimator_list} for j in range(n_dataset)}
  mse = {j: {k: 0 for k in estimator_list}for j in range(n_dataset)}
  rank_cc = {j: {k: 0 for k in estimator_list}for j in range(n_dataset)}
  mean_true_policy_value = np.mean(conventional_metrics_dict[behavior_name][0][estimator_list[0]]['true_policy_value'])

  for i in range(n_dataset):
    for estimator in estimator_list:
      regret[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['regret'][0]/abs(mean_true_policy_value)
      mse[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['mean_squared_error']/(mean_true_policy_value**2)
      rank_cc[i][estimator] = conventional_metrics_dict[behavior_name][i][estimator]['rank_correlation'][0]

    regret_df = pd.DataFrame(regret)
    regret_df['mean'] = regret_df.mean(axis=1)
    regret_df['std'] = regret_df.std(axis=1)

    rank_cc_df = pd.DataFrame(rank_cc)
    rank_cc_df['mean'] = rank_cc_df.mean(axis=1)
    rank_cc_df['std'] = rank_cc_df.std(axis=1)

    mse_df = pd.DataFrame(mse)
    mse_df['mean'] = mse_df.mean(axis=1)
    mse_df['std'] = mse_df.std(axis=1)

    return regret_df, rank_cc_df, mse_df 


