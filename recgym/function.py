
import numpy as np

#実際のuser_transitionの関数、今回はuser_preference_dynamicsを使う
def user_preference_dynamics(state, action, item_feature_vector, alpha = 1):
    state = state + alpha * state @ item_feature_vector[action] * item_feature_vector[action] 
    state = state / np.linalg.norm(state, ord=2)
    return state


#rewardの関数を決定する
def inner_reward_function(state, action, item_feature_vector):
    reward = state @ item_feature_vector[action]
    return reward

# def cos_similar_function(state, action):
#     reward = 
