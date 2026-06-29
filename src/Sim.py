import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import Counter
from copy import deepcopy as cp

from Env import Env
from BaseAgent import BaseAgent
import Policy as pol

def sim(epi, env, agent):
  """シミュレーションの主要部

  param
  -----
  epi(int):
    エピソード数
  param(tuple):
    Agentの学習パラメータ
  Env(Env):
    環境
  Agent(BaseAgent):
    学習Agent

  """
  env = env
  agent = agent
  episode = epi

  count = Counter() #行動選択の偏り確認用カウンター
  countAll = Counter() #sim全体の行動回数カウンター
  steps = [] # ゴール到達までにかかったstep数保存用
  last_path = [] #最終エピソードの経路

  # 描画アニメ用
  fig = plt.figure()
  ims = []

  state_print = False
  for ep in range(episode):
      print(f'episode[{ep}]')
      turn = 0
      term = False
      env.init_episode()
      agent.init_episode()
      if state_print:
          im = env.print_anim()
          ims.append([im])

      #エピソード主要部、終端状態までループ
      while(not term):
          s = env.get_state() # 環境から状態を観測
          countAll[s] += 1
          if state_print:
              last_path.append(s)
              
          a = agent.select_action(s) # エージェントが行動選択
          if state_print:
              print(f'agent select {a}')
          r, sd, term = env.update(a) # 環境を更新(報酬と次状態と終端か否か)
          agent.update(r, sd, term) # Q 値を更新

          count[a] += 1 # 行動選択の偏り確認用

          if state_print:
              print(f'turn:{turn}')
              im = env.print_anim()
              ims.append([im])

          turn += 1
      steps.append(turn)
      if state_print:
        last_path.append(env.get_state())

      # 最後はeval 用
      if ep == episode - 2:
          state_print = True
          agent._behavior_policy = pol.select_greedy

  # gitアニメーションとして保存
  ani = animation.ArtistAnimation(fig, ims, blit=True, interval=1000, repeat_delay=1000)
  ani.save("test.gif", writer='imagemagick')
  agent.print_optimal()


  return steps, countAll, last_path