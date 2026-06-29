import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

import Policy as pol

# エージェントのベースモデル
class BaseAgent(object):

  def __init__(self, env=None, param=(0.1, 0.9, 0.0, 0.01), behavior_policies=None, policy_rate=None, name='Q_eg'):
    up = (0, -1) # 上に行く行動
    down = (0, 1) # 下に行く行動
    left = (-1, 0) # 左に行く行動
    right = (1, 0) # 右に行く行動

    if env is not None:
        self._env = env
        self._width = env.width
        self._height = env.height
    else:
        self._env = None
        self._width = 7
        self._height = 7

    self._last_action = None
    self._update_action = None
    self._alpha = param[0] #学習率
    self._gamma = param[1] #割引率
    self._eps = param[2] # epsilon (探索率初期値)
    # VARIANCE
    self._alpha_v = param[3] #分散用学習率
    self._tau = 0.1 # 温度パラメータ 0に近いほどグリーディー

    self._current_s = None
    self._update_s = None
    self._actions = [up, down, left, right]
    self._Q = {((i, j), action): 0 for i in range(self._width) for j in range(self._height) for action in self._actions}
    self._V = {((i, j), action): 0 for i in range(self._width) for j in range(self._height) for action in self._actions}
    self._count_action = Counter()
    self._pre_r = None

    if behavior_policies is None:
      behavior_policies = [pol.greedy_distribution]

    if policy_rate is None:
      policy_rate = [1.0]
      
    if len(behavior_policies) != len(policy_rate):
       raise ValueError("長さが違う")
    
    rates = np.array(policy_rate)
    rates = rates / rates.sum()

    self._behavior_policies = behavior_policies
    self._policy_rate = rates
    self._estimation_policy = pol.select_greedy

  @property
  def eps(self):
    return self._eps

  @eps.setter
  def eps(self, eps):
    self._eps = eps

  @property
  def tau(self):
    return self._tau

  @tau.setter
  def tau(self, tau):
    self._tau = tau

  def init_episode(self):
      self._current_s = None
      self._update_s = None
      self._last_action = None
      self._update_action = None
      self._pre_r = None

  def _mix_distribution(self, s):
    probs = np.zeros(len(self._actions))

    for policy, rate in zip(self._behavior_policies, self._policy_rate):
        probs += np.array(
            policy(
                self._Q,
                s,
                self._actions,
                self
            )
        ) * rate

    probs /= probs.sum()
    return tuple(probs)
  
  def _mix_select(self, s):
    props = self._mix_distribution(s)
    a = self._actions[np.random.choice(len(self._actions), p=props)]
    return a

  def select_action(self, s):
    """挙動方策によって行動を選択し選択行動を返す"""
    self._current_s = s
    a = self._mix_select(self._current_s)
    self._count_action[s, a] += 1
    self._last_action = a

    return a

  def _update_param(self):
    """探索パラーメタのアップデート"""
    self._eps = 1.0*self._eps
    self._tau = 1.0*self._tau

  def update(self, r, next_s, term):
    pass

  def print_optimal(self):
    """最適方策(Greedy選択)表示用"""
    action = ['UP', 'DN', 'LT', 'RT']
    for y in range(self._height):
        print('|', end='')
        for x in range(self._width):
            print(f'{pol.select_greedy(self._Q, (x, y), self._actions)}\t', end='')
            print('|', end='')
        print()