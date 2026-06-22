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


# main
episode = 200
env = Env()
agent = OffPolicyAgent(env.width, env.height, param=(0.1, 0.9, 0.1), behavior_policies=[pol], policy_rate=[1],  name='UCB')

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