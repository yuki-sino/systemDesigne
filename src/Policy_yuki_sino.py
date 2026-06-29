# VARIANCE
import numpy as np

def mv(Q, state, actions, agent=None):
    alpha = 1
    tau = 0.5
    Qvalues = np.array([Q[state, action] for action in actions])
    Vvalues = np.array([agent._V[state, action] for action in actions])
    scores = Qvalues - alpha*Vvalues

    scores -= np.max(scores)

    probs = np.exp(scores/ tau)
    probs /= np.sum(probs)
    return probs[0], probs[1], probs[2], probs[3]

def mv_selectRisk(Q, state, actions, agent=None):
    alpha = 1
    tau = 0.2
    Qvalues = np.array([Q[state, action] for action in actions])
    Vvalues = np.array([agent._V[state, action] for action in actions])
    scores = Qvalues + alpha*Vvalues

    scores -= np.max(scores)

    probs = np.exp(scores/ tau)
    probs /= np.sum(probs)
    return probs[0], probs[1], probs[2], probs[3]