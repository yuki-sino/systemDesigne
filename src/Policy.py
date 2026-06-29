import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp
import math

import Policy_24rd172
import Policy_yuki_sino

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

def e_greedy_distribution(Q, state, actions, agent):
  eps = agent.eps
  if random.random() < eps:
    action_distribution = random_distribution(Q, state, actions, agent)
  else:
    action_distribution = greedy_distribution(Q, state, actions, agent)
  return action_distribution

def softmax_distribution(Q, state, actions, agent):
  tau = agent.tau
  Qvalues = np.array([Q[state, action] for action in actions])
  p_boltzmann = dist_boltzmann(Qvalues, tau)
  return p_boltzmann[0], p_boltzmann[1], p_boltzmann[2], p_boltzmann[3]

def dist_boltzmann(Q, tau):
  """Boltzmann 分布　= Softmax変換関数"""
  trQ = np.exp(Q - np.max(Q))/tau
  return trQ / trQ.sum(axis=0)

def ucb_distribution(Q, state, actions, agent):
  return Policy_24rd172.ucb_distribution(Q, state, actions, agent)

def mv(Q, state, actions, agent):
  return Policy_yuki_sino.mv(Q, state, actions, agent)

def mv_selectRisk(Q, state, actions, agent=None):
  return Policy_yuki_sino.mv_selectRisk(Q, state, actions, agent)
