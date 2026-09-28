"""REINFORCE Policy Gradient with Baseline Engine.
100% Python Standard Library.
"""

import math
import random

class REINFORCEPolicyGradient:
    """REINFORCE Monte Carlo policy gradient with state-value baseline subtraction."""
    def __init__(self, num_states=4, num_actions=2, lr=0.1, gamma=0.99):
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.theta = [[0.0] * num_actions for _ in range(num_states)]
        self.baseline = [0.0] * num_states

    def get_action_probs(self, state):
        logits = self.theta[state]
        max_l = max(logits)
        exp_l = [math.exp(l - max_l) for l in logits]
        sum_exp = sum(exp_l)
        return [e / sum_exp for e in exp_l]

    def sample_action(self, state):
        probs = self.get_action_probs(state)
        r = random.random()
        cum = 0.0
        for a, p in enumerate(probs):
            cum += p
            if r <= cum:
                return a
        return len(probs) - 1

    def update_trajectory(self, trajectory):
        T = len(trajectory)
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + self.gamma * G
            returns[t] = G

        for t in range(T):
            s, a, _ = trajectory[t]
            G_t = returns[t]
            b_s = self.baseline[s]
            advantage = G_t - b_s
            self.baseline[s] += 0.1 * advantage

            probs = self.get_action_probs(s)
            for act in range(self.num_actions):
                grad = (1.0 - probs[act]) if act == a else (-probs[act])
                self.theta[s][act] += self.lr * grad * advantage
