import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

import Policy as pol

# エージェントのベースモデル
## 継承して update 関数を実装して使う
class BaseAgent(object):
  """glid world Agent class

  param
  -----
  width(int): default: 7
    glid width
  height(int): default: 7
    glid height
  param(tuple): default: (0.1, 0.9, 0)
    RL learning parameters: alpha, gammma, epsilon
  policy(string): default: Q
    RL Policy: Q epsilon greedy, Q softmax, sarsa epsilon greedy, sarsa softmax
    Q: policy off(behavior != estimation)
    sarsa: policy on(behavior == estimation)
  R(float): default: 0.99
    garbage

  attr
  -----
  _last_action(tuple(int, int)):
    最後に行った行動、学習時に用いる
  _alpha(float):
    学習率
  _gammma(float):
    減衰率
  _eps(float):
    εGreedy時のランダム行動しきい値
  _current_s(tuple(int, int)):
    現在の状態、今回の場合中身はエージェントの位置座標が保存されている(他に変化する情報がないため)
  _action(list(tuple)):
    行動リスト、エージェントの行動として、x座標とy座標に加算する値のTupleが上下左右として入っている
  _Q(dict(float):
    Q値、状態行動対の文字列をキー、それに対応する価値をvalueとし、0で初期化
  _count_action(Counter):
    各行動をどれくらい行ったかをCounterでカウント、使途はない
  _behavior_policy(function):
    挙動方策に何を用いるか、soft方策(各状態を十全に探索できる)である必要がある
  _estimation_policy(function):
    推定方策に何を用いるか
  """
  def __init__(self, width=7, height=7, param=(0.1, 0.9, 0.0), behavior_policies=None, policy_rate=None, name='Q_eg'):
    up = (0, -1) # 上に行く行動
    down = (0, 1) # 下に行く行動
    left = (-1, 0) # 左に行く行動
    right = (1, 0) # 右に行く行動

    self._height = height
    self._width = width

    self._last_action = None
    self._update_action = None
    self._alpha = param[0] #学習率
    self._gamma = param[1] #割引率
    self._eps = param[2] # epsilon (探索率初期値)
    self._tau = 1.0 # 温度パラメータ

    self._current_s = None
    self._update_s = None
    self._actions = [up, down, left, right]
    self._Q = {((i, j), action): 0 for i in range(width) for j in range(height) for action in self._actions}
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
    # 学習アルゴリズムによって方策を切り替える
    # self._behavior_policy = behavior_policies # 実際に実行した行動が格納されている
    self._estimation_policy = pol.select_greedy # 最大のQ値の行動が格納されている

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

      # self._behavior_policy

  def _mix_distribution(self, s):
    probs = np.zeros(len(self._actions))

    for policy, rate in zip(self._behavior_policies, self._policy_rate):
        probs += np.array(
            policy(
                self._Q,
                s,
                self._actions,
                self._eps
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
    # a = self._behavior_policy(self._Q, self._current_s, self._actions, self._eps)
    a = self._mix_select(self._current_s)
    self._count_action[s, a] += 1
    self._last_action = a

    return a

  def _update_param(self):
    """探索パラーメタのアップデート"""
    self._eps = 1.0*self._eps
    self._tau = 1.0*self._tau

  def update(self, r, next_s, term):
    # 継承して実装
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