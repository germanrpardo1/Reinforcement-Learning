# Reinforcement-Learning
Reinforcement learning course taken during my MSc degree at LSE.

For the final project we implemented a variety of models including Q-Learning, Double Q-Learning, Neural Fitted Q-Iteration, Deep Q-Network, Monte Carlo Tree search (with exploring starts) and AlphaGo Zero (with some variants and exploring starts) to solve the nxn multiple boards Tic-Tac-Toe which complexity is larger than the usual game. Our best model was that of Monte Carlo Tree Search, it learns very fast compared to the others and outperforms all of the other algorithms in just a short amount of training time. However, it was left for futur work to train for longer time the AlphaGo Zero algorithm.


Topics of the course include:

1. Introduction to RL.
2. Foundations of RL: Markov decision processes, Bellman optimality equation, the existence of optimal stationary policy. Multi-Armed Bandits, Optimistic Principle.
3. Dynamic programing and Monte Carlo methods: policy evaluation, policy improvement, policy iteration, value iteration based on dynamic programming, and Monte Carlo methods for RL, including Monte Carlo estimation and Monte Carlo control.
4. Temporal difference learning: temporal difference learning, temporal difference prediction, SARSA, Q-Learning and n-step temporal difference predictions, TD(lambda).
5. On-policy prediction and control with approximation: types of function approximators (value and action-value function approximator), gradient based methods for value function prediction, convergence guarantees with linear function approximator, and semi-gradient n-step SARSA. 
6. Q-Learning type algorithms with function approximation: Q-Learning with linear function approximator, Fitted Q-Iteration, Deep Q-Network.
7. Policy gradient methods: policy approximation, REINFORCE, actor-critic methods that combine policy function approximation with action-value function approximation.
8. Model-Based Learning: Dyna-Q, Monte Carlo Tree Search, AlphaGo.
9. Batch off-policy evaluation: importance sampling-based method, doubly robust method, marginalized importance sampling.
10. Batch policy optimisation: recent advances in offline RL algorithms.
