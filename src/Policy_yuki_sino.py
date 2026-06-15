import numpy as np
import random

# 現在位置s
# w, a, x, d方向がある。それぞれの方向を進んだ後に進める方向がある。
# 各方向を含めた、各方向を進んだ後に進めるマスのs以外の平均、分散を計算してそれを使う

def Mean_Variance_Objective(Q, state, actions, dummy=None):
    