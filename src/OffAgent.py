import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from BaseAgent import BaseAgent

class OffPolicyAgent(BaseAgent):
  """ Off-policy Agetn = Q-learning Agent """

  def _update_param(self):
    """探索パラーメタのアップデート"""
    self._eps = 1.0*self._eps
    self._tau = 1.0*self._tau

  def update(self, r, next_s, term):
    """Q値更新関数
    r(float):
      報酬
    sd(tuple(int, int)):
      次状態
    term(bool):
      終端状態判別フラグ

    """
    buckup_action = self._estimation_policy(self._Q, self._current_s, self._actions, self._eps)


    if self._update_s is not None:
        target = self._pre_r + self._gamma * self._Q[(self._current_s, buckup_action)] - self._Q[(self._update_s, self._update_action)]
        self._Q[(self._update_s, self._update_action)] += self._alpha * target

    if term:
        self._Q[(self._current_s, self._last_action)] += self._alpha * r
        self._update_param() # 終端についたら探索パラメータを更新 (ここでなくてもよい)
    else:
        self._update_s = cp(self._current_s)
        self._update_action = cp(self._last_action)
        self._current_s = next_s
        self._pre_r = cp(r)