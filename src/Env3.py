import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from Env2 import Env2

class Env3(Env2):
    s = 1 #スタート
    g = 2 #ゴール
    c = 3 #崖
    w = 4 #壁
    # 5~9：大きくなるほど分散が小さい報酬、期待値は0

    def __init__(self, post_coordinates=(0, 0), pos_goal=(13, 9)):
        self._size = {'width': 14, 'height': 10}
        self._state = [[0 for _ in range(self._size['height'])] for _ in range(self._size['width'])]

        self._coordinates = post_coordinates
        self._s = post_coordinates
        self._goal = pos_goal

        self._state=[
            [1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [3,3,0,3,3,3,3,0,0,0],
            [0,0,0,0,0,3,3,0,0,0],
            [3,3,3,3,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0],
            [0,9,8,7,6,5,4,4,4,4],
            [0,9,8,7,6,5,4,0,0,0],
            [0,9,8,7,6,5,0,0,0,0],
            [0,0,0,3,3,0,4,4,4,0],
            [0,0,0,3,3,0,4,0,0,0],
            [0,0,0,0,0,0,4,0,4,4],
            [0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,2],
        ]

        self._stateBackUp = cp(self._state)

    def init_episode(self):
        coordinate_x, coordinate_y = self._s
        gl_x, gl_y = self._goal
        self._coordinates = self._s

        self._state = cp(self._stateBackUp)


    def update(self, action):
        dif_x, dif_y = action
        coordinate_x, coordinate_y = self._coordinates

        r = 0
        term = False
        self._state[coordinate_x][coordinate_y] = 0

        # 次の座標を計算
        next_x = self._limit_pos(coordinate_x + dif_x, 'width')
        next_y = self._limit_pos(coordinate_y + dif_y, 'height')


        # ゴール判定
        if self._state[next_x][next_y] == 2:
            r = 1
            term = True
        # 崖判定
        elif self._state[next_x][next_y] == 3:
            r = -100
            next_x, next_y = self._s
        # 壁にぶつかった際の判定
        elif self._state[next_x][next_y] == 4:
            # 壁にぶつかったら移動できず、元の位置に留まる
            next_x, next_y = coordinate_x, coordinate_y
            r = -1

        # 分散のマス
        elif self._state[next_x][next_y] == 5:
            score = 16
            if random.random() < 0.5:
                r = score
            else:
                r = -1 * score - 2
        elif self._state[next_x][next_y] == 6:
            score = 8
            if random.random() < 0.5:
                r = score
            else:
                r = -1 * score - 2
        elif self._state[next_x][next_y] == 7:
            score = 4
            if random.random() < 0.5:
                r = score
            else:
                r = -1 * score - 2
        elif self._state[next_x][next_y] == 8:
            score = 2
            if random.random() < 0.5:
                r = score
            else:
                r = -1 * score - 2
        elif self._state[next_x][next_y] == 9:
            score = 1
            if random.random() < 0.5:
                r = score
            else:
                r = -1 * score - 2
        # 通常の道
        else:
            r = -1

        if not(coordinate_x - next_x == 0 and coordinate_y - next_y == 0):
            self._state[coordinate_x][coordinate_y] = cp(self._stateBackUp[coordinate_x][coordinate_y])

        self._coordinates = (next_x, next_y)
        self._state[next_x][next_y] = 1

        return r, (next_x, next_y), term