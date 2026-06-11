import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

# 崖歩き課題環境を作成
class Env(object):
  """Glid World Environment class

  param
  -----
  post_coordinates(tuple(int, int)):
    Agent 初期位置
  pos_goal(tuple(int, int)):
    ゴール位置

  attr
  -----
  _size(dict):
    glid サイズ
  _state(list(list(int))):
    二次元list、状態を管理
  _coordinates(tuple(int, int)):
    Environment上のAgentの位置 （座標）
  _s(tuple(int, int)):
    現在の状態 (Agentの位置)
  _goal(tuple(int, int)):
    ゴールの位置
  """
  def __init__(self, post_coordinates=(0, 7), pos_goal=(11, 7)):
    self._size = {'width': 12, 'height': 8}
    self._state = [[0 for _ in range(self._size['height'])] for _ in range(self._size['width'])]

    self._coordinates = post_coordinates
    self._s = post_coordinates
    self._goal = pos_goal

  def init_episode(self):
    """Envの episode ごとのリセット関数"""
    coordinate_x, coordinate_y = self._s
    gl_x, gl_y = self._goal
    self._coordinates = self._s
    self._state[coordinate_x][coordinate_y] = 1
    self._state[gl_x][gl_y] = 9

    for i in range(1, 11):
      self._state[i][7] = 5

  @property
  def width(self):
    return self._size['width']

  @property
  def height(self):
    return self._size['height']

  def _limit_pos(self, pos, direct):
    """Agentの移動範囲制御"""
    if pos < 0:
        return 0
    if pos > self._size[direct] - 1:
        return self._size[direct] - 1

    return pos
  def get_state(self):
    """現在状態の取得"""
    return self._coordinates

  def update(self, action):
    """Environmentの更新
    学習Agent が選択した行動を反映し、報酬と次状態を返す
    """
    dif_x, dif_y = action
    coordinate_x, coordinate_y = self._coordinates

    r = 0
    term = False
    self._state[coordinate_x][coordinate_y] = 0

    coordinate_x = self._limit_pos(coordinate_x + dif_x, 'width')
    coordinate_y = self._limit_pos(coordinate_y + dif_y, 'height')

    s_post = (coordinate_x, coordinate_y)

    #ゴールに到達した時にの報酬
    if self._state[coordinate_x][coordinate_y] == 9:
      r = 1
      term = True
    elif self._state[coordinate_x][coordinate_y] == 5:
      r = -10000
      coordinate_x, coordinate_y = self._s
    else:
      r = -1

    self._coordinates = (coordinate_x, coordinate_y)
    self._state[coordinate_x][coordinate_y] = 1

    return r, s_post, term

  def print_anim(self):
    """画像の生成用"""
    plt.xlim(0, self.width)
    plt.ylim(self.height, 0)
    rows = np.arange(self.width+ 1)
    cols = np.arange(self.height + 1)
    X, Y = np.meshgrid(rows, cols)
    C = list(map(list, zip(*self._state)))
    im = plt.pcolor(X, Y, C, cmap=plt.cm.gray_r)
    plt.grid(True, which='both', axis='both', linestyle='-', color='k')
    plt.show()
    return im