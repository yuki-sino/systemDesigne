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
import Policy_24rd172 as pol2


# main
episode = 400
env = Env2()
agent = OffPolicyAgent(env, param=(0.1, 0.9, 0.1), behavior_policies=[pol.softmax_distribution,pol2.wall_avoidance_distribution], policy_rate=[0.9,0.1],  name='UCB')

steps_Q_e_greedy = sim(epi=episode, env=env, agent=agent)

# 結果
plt.figure(figsize=(12.0, 6.0))
plt.plot(np.arange(episode), steps_Q_e_greedy)
plt.legend(
    ['Q_e_greedy',
     ],
    loc='upper right'
)
plt.show()
print('last policy steps')
print('  Q_e_greedy: \t{} steps'.format(steps_Q_e_greedy[-1]))