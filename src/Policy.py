import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp
import math

# 行動方策一覧
def select_greedy(Q, state, actions, dummy=None):
  """Greedy選択関数"""
  Qvalues = np.array([Q[state, action] for action in actions])
  idx = np.where(Qvalues == Qvalues.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]
  return action_selected

# ランダムに選ぶ分布
# 引数の4つ目を一律 agent に変更（デフォルト値を None にしておくと便利）
def random_distribution(Q, state, actions, agent=None):
  return 1/4, 1/4, 1/4, 1/4

def greedy_distribution(Q, state, actions, agent=None):
  Qvalues = np.array([Q[state, action] for action in actions])
  idx = np.where(Qvalues == Qvalues.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]
  actionprop = [1 if action == action_selected else 0 for action in actions]
  return actionprop[0], actionprop[1], actionprop[2], actionprop[3]

# eps を agent から取得する
def e_greedy_distribution(Q, state, actions, agent):
  eps = agent.eps  # agent のプロパティから取得
  if random.random() < eps:
    action_distribution = random_distribution(Q, state, actions, agent)
  else:
    action_distribution = greedy_distribution(Q, state, actions, agent)
  return action_distribution

# tau を agent から取得する
def softmax_distribution(Q, state, actions, agent):
  tau = agent.tau  # agent のプロパティから取得
  Qvalues = np.array([Q[state, action] for action in actions])
  p_boltzmann = dist_boltzmann(Qvalues, tau)
  return p_boltzmann[0], p_boltzmann[1], p_boltzmann[2], p_boltzmann[3]

# カウント情報やハイパーパラメータを agent から取得する
def ucb_distribution(Q, state, actions, agent):
  c = agent.eps
  count_action = agent._count_action
  
  total_visit = sum(count_action[state, a] for a in actions)
  
  if total_visit == 0:
    return 1/4, 1/4, 1/4, 1/4

  ucb_values = []
  for action in actions:
    n_sa = count_action[state, action]
    if n_sa == 0:
      ucb_val = float('inf')
    else:
      ucb_val = Q[state, action] + c * math.sqrt(math.log(total_visit) / n_sa)
    ucb_values.append(ucb_val)
    
  ucb_values = np.array(ucb_values)

  idx = np.where(ucb_values == ucb_values.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]

  actionprop = [1 if action == action_selected else 0 for action in actions]
  return actionprop[0], actionprop[1], actionprop[2], actionprop[3]