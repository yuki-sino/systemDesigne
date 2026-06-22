import numpy as np
import random

def Mean_Variance_Objective(Q, state, actions, agent):
    # lam = agent.lam
    lam = 0.9

    x, y = state
    riskmap = agent.riskmap  # (p, r, v) が入っている前提
    xlim = len(riskmap)
    ylim = len(riskmap[0])
    mv_values = []

    for action in actions:
        dx, dy = action
        nx, ny = x + dx, y + dy

        q_val = Q[state, action]

        if nx >= xlim or ny >= ylim:
            mv = q_val
        else:
            p, r, v = riskmap[nx][ny]
            fear_bonus = p * r - lam * p * v 
            # mv = p * r - lam * p * v
            mv = q_val + fear_bonus
        mv_values.append(mv)

    mv_values = np.array(mv_values)

    mv_max = np.max(mv_values)
    exp_vals = np.exp(mv_values - mv_max)
    probs = exp_vals / np.sum(exp_vals)

    return probs[0], probs[1], probs[2], probs[3]    