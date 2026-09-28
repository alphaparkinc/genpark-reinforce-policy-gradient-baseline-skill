from client import REINFORCEPolicyGradient

agent = REINFORCEPolicyGradient(num_states=2, num_actions=2, lr=0.2)
for _ in range(30):
    agent.update_trajectory([(0, 1, 10.0), (0, 0, 0.0)])

probs = agent.get_action_probs(0)
print(f"Policy Action Probabilities in State 0: Action 0={probs[0]:.4f}, Action 1={probs[1]:.4f}")
