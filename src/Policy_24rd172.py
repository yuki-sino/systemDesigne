import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp
import math

def ucb_distribution(Q, state, actions, agent):
  c = agent.eps
  count_action = agent._count_action
  
  total_visit = sum(count_action[state, a] for a in actions)
  
  if total_visit == 0:
    return 1/4, 1/4, 1/4, 1/4

  ucb_values = []
  for action in actions:
    n_sa = count_action[state, action]
    if n_sa == 0:
      ucb_val = float('inf')
    else:
      ucb_val = Q[state, action] + c * math.sqrt(math.log(total_visit) / n_sa)
    ucb_values.append(ucb_val)
    
  ucb_values = np.array(ucb_values)

  idx = np.where(ucb_values == ucb_values.max())
  if len(idx[0]) > 1:
    action_selected = actions[random.choice(idx[0])]
  else:
    action_selected = actions[idx[0][0]]

  actionprop = [1 if action == action_selected else 0 for action in actions]
  return actionprop[0], actionprop[1], actionprop[2], actionprop[3]