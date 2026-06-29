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
  return tuple(action_selected)

def reverse_avoidance_distribution(Q, state, actions, agent):

  last_action = agent._last_action

  if last_action is None:
    return 1/4, 1/4, 1/4, 1/4

  reverse_action = (-last_action[0], -last_action[1])
  
  weights = []
  for action in actions:
    if action == reverse_action:
      weights.append(0.1)
    elif action == last_action:
      weights.append(1.5)
    else:
      weights.append(1.0)
      
  weights = np.array(weights)
  probs = weights / weights.sum()
  
  return tuple(probs)

def wall_avoidance_distribution(Q, state, actions, agent):
    x, y = state
    w, h = agent._width, agent._height
    
    if agent._env is None:
        return 1/4, 1/4, 1/4, 1/4
        
    valid_weights = []
    for action in actions:
        nx, ny = x + action[0], y + action[1]
        
        if not (0 <= nx < w and 0 <= ny < h):
            valid_weights.append(0.0)
            continue
            
        if agent._env._state[nx][ny] == 8:
            valid_weights.append(0.0)
            continue
            
        valid_weights.append(1.0)
        
    weights = np.array(valid_weights)
    if weights.sum() == 0:
        return 1/4, 1/4, 1/4, 1/4
        
    probs = weights / weights.sum()
    return tuple(probs)