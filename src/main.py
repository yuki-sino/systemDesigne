import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from Env import Env
from OffAgent import OffPolicyAgent
from Sim import sim
import Policy as pol
from Env3 import Env3


# main
episode = 500
# env = Env()
env = Env3()
agent = OffPolicyAgent(env.width, env.height, param=(0.1, 0.9, 0.1, 0.01), behavior_policies=[pol.greedy_distribution, pol.mv_selectRisk], policy_rate=[2,1],  name='UCB')

steps_Q_e_greedy, countAll = sim(epi=episode, env=env, agent=agent)

# 結果
plt.figure(figsize=(12.0, 6.0))
plt.plot(np.arange(episode), steps_Q_e_greedy)
plt.legend(
    ['Q_e_greedy',
     ],
    loc='upper right'
)
plt.show()

visit_map = np.zeros((env.width, env.height))
for (x, y), cnt in countAll.items():
    visit_map[x][y] = cnt

rows = np.arange(env.width + 1)
cols = np.arange(env.height + 1)
X, Y = np.meshgrid(rows, cols)

visit_map = np.log1p(visit_map)
C = list(map(list, zip(*visit_map)))

plt.figure()
# plt.pcolor(X, Y, C, cmap="Reds")
# plt.grid(True)
plt.pcolor(X, Y, C, cmap="Reds")
plt.ylim(env.height, 0)
plt.grid(True)
plt.colorbar(label="Visit Count")
plt.show()


print('last policy steps')
print('  Q_e_greedy: \t{} steps'.format(steps_Q_e_greedy[-1]))