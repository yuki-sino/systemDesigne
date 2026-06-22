import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from Env import Env

class Env2(Env):

    def __init__(self, post_coordinates=(0, 0), pos_goal=(13, 9)):
        self._size = {'width': 14, 'height': 10}
        self._state = [[0 for _ in range(self._size['height'])] for _ in range(self._size['width'])]

        self._coordinates = post_coordinates
        self._s = post_coordinates
        self._goal = pos_goal

    def init_episode(self):
        coordinate_x, coordinate_y = self._s
        gl_x, gl_y = self._goal
        self._coordinates = self._s

        self._state=[
            [1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [5,5,0,5,5,5,5,0,0,0],
            [0,0,0,0,0,5,5,0,0,0],
            [5,5,5,5,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,8,8,8,8],
            [0,0,0,0,0,0,8,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,5,5,0,8,8,8,0],
            [0,0,0,5,5,0,8,0,0,0],
            [0,0,0,0,0,0,8,0,8,8],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,9],
        ]

    def update(self, action):
        dif_x, dif_y = action
        coordinate_x, coordinate_y = self._coordinates

        r = 0
        term = False
        self._state[coordinate_x][coordinate_y] = 0

        # 次の座標を計算
        next_x = self._limit_pos(coordinate_x + dif_x, 'width')
        next_y = self._limit_pos(coordinate_y + dif_y, 'height')

        # 壁にぶつかった際の判定
        if self._state[next_x][next_y] == 8:
            # 壁にぶつかったら移動できず、元の位置に留まる
            next_x, next_y = coordinate_x, coordinate_y
            r = -1

        # ゴール判定
        elif self._state[next_x][next_y] == 9:
            r = 1
            term = True
        # 崖判定
        elif self._state[next_x][next_y] == 5:
            r = -100
            next_x, next_y = self._s
        # 通常の道
        else:
            r = -1

        self._coordinates = (next_x, next_y)
        self._state[next_x][next_y] = 1

        return r, (next_x, next_y), term