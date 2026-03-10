# SC3000 LAB ASSIGNMENT 1 

# PART 1: FINDING A SHORTEST PATH WITH AN ENERGY BUDGET 

<!-- task1  -->
<!-- task2  -->
<!-- task3  -->


# PART 2: SOLVING MDP AND REINFORCEMENT LEARNING PROBLEMS USING A GRID WORLD

The Part 2 of this project focuses on guiding an agent to navigate a maze-like grid to reach its goal state while avoiding roadblocks and minimizing the cost. The environment is stochastic (actions do not always go exactly where it is intended). The main object is to find the policy. 

## Environment 
Grid size: 5 x 5
Start State: (0,0)
Goal State: (4,4)

Roadblocks: (2,1) ,(2,3)

Actions: UP, DOWN, LEFT, RIGHT 

Transition Model: 
- Intended direction: 0.8 
- Perpendicular directions: 0.1 each 

Rewards: 
- Step Reward: -1
- Goal Reward: +10 

## Tasks Implemented 

### Task 1: Value Iteration and Policy Iteration 
Here the agent knows the transition model. 
Two algorithms implemented:
#### Value Iteration 
Computes optimal value for each state and derives best action from those values.
#### Policy Iteration 
Starts with random policy and repeatedly evaluates the policy and improves the policy until the optimal policy is found 

### Task 2: Monte Carlo Control 
Here the agent does not know the transition probabilities. It learns by experience.  

1. Runs many episodes from the start state 
2. Observes the rewards 
3. Estimates state-action values using sampled returns 
4. Improves policy using E-greedy exploration 

### Task 3: Q-Learning 

Q-Learning updates after every step and is a fast learning process. It uses temporal difference learning. 

Q(s,a)=Q(s,a)+α[r+γmaxQ(s′,a)−Q(s,a)]

The agent gradually learns the optimal policy through interaction with the environment




