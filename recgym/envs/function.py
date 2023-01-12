
import numpy as np

#user_transition_function
# user_feature is amplified by the recommended item_feature
def user_preference_dynamics(state, action, item_feature_vector, alpha = 1):
    state = state + alpha * state @ item_feature_vector[action] * item_feature_vector[action] 
    state = state / np.linalg.norm(state, ord=2)
    return state

#reward_function
# inner product of state and recommended item_feature
def inner_reward_function(state, action, item_feature_vector):
    reward = state @ item_feature_vector[action]
    return reward


