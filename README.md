# genpark-reinforce-policy-gradient-baseline-skill

Agent Skill implementing the **REINFORCE Policy Gradient Algorithm** with state-value baseline subtraction for variance reduction in episodic reinforcement learning.

## Architectural Overview
```mermaid
flowchart TD
    Rollout["Episodic Trajectory (s_t, a_t, r_t)"] --> Returns["Compute Discounted Returns G_t = sum gamma^k r_{t+k}"]
    Returns --> Adv["Compute Advantage: A_t = G_t - b(s_t)"]
    Adv --> Grad["Policy Score Function Grad log pi(a_t|s_t)"]
    Grad --> Update["Parameter Ascent: theta += alpha * Grad * A_t"]
    Update --> Baseline["Update Baseline Tracker b(s_t)"]
```
