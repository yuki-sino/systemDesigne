import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

# 行動方策一覧

# ランダムに選ぶ方策
def select_random(Q, state, actions, dummy=None):
  action_selected = random.choice(actions)
  return action_selected

# 必ず最も高いQ値の高い行動を選ぶ方策
def select_greedy(Q, state, actions, dummy=None):
  """Greedy選択関数"""
  Qvalues = np.array([Q[state, action] for action in actions])
  idx = np.where(Qvalues == Qvalues.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]
  return action_selected

# epsilon-greedy 方策
def e_greedy(Q, state, actions, eps=0.1):
  """εGreedy選択、εをしきい値として活用(Greedyな行動)と探索(ランダム行動)を切り替え、行動を返す"""
  if random.random() < eps:
      # 行動をランダム選択
      action_selected = select_random(Q, state, actions)
  else:
    action_selected = select_greedy(Q, state, actions)
  return action_selected

# softmax 方策
def softmax(Q, state, actions, tau=1.0):
  """Softmax選択"""
  Qvalues = np.array([Q[state, action] for action in actions])
  p_boltzmann = dist_boltzmann(Qvalues, tau)
  a = actions[np.random.choice(len(actions), p=p_boltzmann)]
  return a

# softmax 方策に必要なQ値と温度パラメータtauに応じたboltzmann分布を生成する関数
def dist_boltzmann(Q, tau):
  """Boltzmann 分布　= Softmax変換関数"""
  trQ = np.exp(Q - np.max(Q))/tau
  return trQ / trQ.sum(axis=0)

# ランダムに選ぶ分布
def random_distribution(Q, state, actions, dummy=None):
  up = 1/4
  down = 1/4
  left = 1/4
  right = 1/4
  return up, down, left, right


# 必ず最も高いQ値の高い行動を選ぶ分布
def greedy_distribution(Q, state, actions, dummy=None):
  """Greedy選択関数"""
  Qvalues = np.array([Q[state, action] for action in actions])

  idx = np.where(Qvalues == Qvalues.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]

  actionprop = [
    1 if action == action_selected else 0
    for action in actions
  ]
  
  return actionprop[0], actionprop[1], actionprop[2], actionprop[3]

# epsilon-greedy 方策
def e_greedy_distribution(Q, state, actions, eps=0.1):
  """εGreedy選択、εをしきい値として活用(Greedyな行動)と探索(ランダム行動)を切り替え、行動を返す"""
  if random.random() < eps:
    # 行動をランダム選択
    action_distribution = random_distribution(Q, state, actions)
  else:
    action_distribution = greedy_distribution(Q, state, actions)
  return action_distribution

# softmax分布
def softmax_distribution(Q, state, actions, tau=1.0):
  """Softmax選択"""
  Qvalues = np.array([Q[state, action] for action in actions])
  p_boltzmann = dist_boltzmann(Qvalues, tau)
  return p_boltzmann[0], p_boltzmann[1], p_boltzmann[2], p_boltzmann[3]

# softmax 方策に必要なQ値と温度パラメータtauに応じたboltzmann分布を生成する関数
def dist_boltzmann(Q, tau):
  """Boltzmann 分布　= Softmax変換関数"""
  trQ = np.exp(Q - np.max(Q))/tau
  return trQ / trQ.sum(axis=0)

