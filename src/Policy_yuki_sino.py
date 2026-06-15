import numpy as np
import random
import Policy as pol

def future_prediction(Q, state, actions, d=3, behavior_policies=None, policy_rates=None):
    """
    現時点の位置から、上下左右に進んだ後に行動方策に従ってdステップ進むことで
    得られる方策の値が最も高い方向を出す方策
    未来予測には混合方策（複数の行動方策の確率的な組み合わせ）を使用します。
    """
    if behavior_policies is None:
        behavior_policies = [pol.select_greedy]
    if policy_rates is None:
        policy_rates = [1.0]

    # Qテーブルのキーからマップの最大座標を取得
    max_x = max([s[0] for s, a in Q.keys()])
    max_y = max([s[1] for s, a in Q.keys()])

    def get_next_state(s, a):
        nx = max(0, min(max_x, s[0] + a[0]))
        ny = max(0, min(max_y, s[1] + a[1]))
        return (nx, ny)

    def select_action_mixed(s_current):
        # 確率分布に従って方策を一つ選ぶ
        policy_idx = np.random.choice(len(behavior_policies), p=policy_rates)
        selected_policy = behavior_policies[policy_idx]
        return selected_policy(Q, s_current, actions)

    best_value = -float('inf')
    best_actions = []

    for a in actions:
        current_s = get_next_state(state, a)
        path_value = Q[state, a]
        
        # 行動方策に従ってdステップ進む
        for _ in range(d):
            next_a = select_action_mixed(current_s)
            path_value += Q[current_s, next_a]
            current_s = get_next_state(current_s, next_a)

        if path_value > best_value:
            best_value = path_value
            best_actions = [a]
        elif path_value == best_value:
            best_actions.append(a)

    return random.choice(best_actions)
