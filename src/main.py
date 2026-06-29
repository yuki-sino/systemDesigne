import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from Env import Env
from Env2 import Env2
from OffAgent import OffPolicyAgent
from Sim import sim
import Policy as pol
from Env3 import Env3


# main
episode = 1000
# env = Env()
env = Env3()
agent = OffPolicyAgent(env.width, env.height, param=(0.1, 0.9, 0.1, 0.01), behavior_policies=[pol.greedy_distribution, pol.softmax_distribution, pol.mv, pol.mv_selectRisk, pol.ucb_distribution], policy_rate=[0,1,0.3,0, 0],  name='UCB')

steps_Q_e_greedy, countAll, last_path = sim(epi=episode, env=env, agent=agent)

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

visit_map = np.log1p(visit_map)

env.imaging(visit_map)
env.draw_border(env._state, env._color_table)
env.draw_path(last_path)
plt.show()

q_map = np.zeros((env.width, env.height))

for x in range(env.width):
    for y in range(env.height):
        q_map[x][y] = max(
            agent._Q[((x, y), a)]
            for a in agent._actions
        )
q_map = np.log1p(q_map)
env.imaging(q_map)
env.draw_border(env._state, env._color_table)
env.draw_path(last_path)
plt.show()

v_map = np.zeros((env.width, env.height))

for x in range(env.width):
    for y in range(env.height):
        v_map[x][y] = max(
            agent._V[((x, y), a)]
            for a in agent._actions
        )
v_map = np.log1p(v_map)
env.imaging(v_map)
env.draw_border(env._state, env._color_table)
env.draw_path(last_path)
plt.show()
print('last policy steps')
print('  Q_e_greedy: \t{} steps'.format(steps_Q_e_greedy[-1]))