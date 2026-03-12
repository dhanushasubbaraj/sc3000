import numpy as np 
import random as ran

# GRID WORLD 

# Environment set-up
class GridWorldenv:

    # Constructor of the environment 
    def __init__(self):
        self.grid_size = 5  # Grid size 
        self.start = (0,0)  # SStart coordinate
        self.end = (4,4)    # End coordinate 
        self.blockers = [(2,1),(2,3)]   # Roadblocks 
        self.agent_actions = ["UP","DOWN","LEFT","RIGHT"]   # four actions taken by the agent
        self.gamma_val = 0.9 # discount factor 

    # Validates the state 
    def state_check(self,state):
        x,y = state                 # extracts coordinates
        # Prevents agent from leaving the grid
        if x<0 or x>=self.grid_size:
            return False 
        if y<0 or y>=self.grid_size:
            return False
        # Checks for presence of obstacles
        if state in self.blockers:
            return False
        return True     
    
    # Computes next state values in accordance to the action taken 
    def directions(self,state,dirs):
        x,y = state                # extracts coordinates

        # Movement of the agent 
        if dirs == "UP":
            upd_state = (x,y+1)

        elif dirs == "DOWN":
            upd_state = (x,y-1)

        elif dirs == "RIGHT":
            upd_state = (x+1,y)

        elif dirs == "LEFT":
            upd_state = (x-1,y)   

        # checks if movements hits roadblockers - agent stays in the same state 
        if not self.state_check(upd_state):
            return state 
        
        return upd_state    # returns new state

    ## TASK 1 
    # TRANSITION MODEL - computes the probability of reaching next state from the current state after taking action 
    def transition_model(self,state,dirs):
        if state == self.end:       # Terminal State -  reached goal 
            return [(state,1,0)]
        transition_dir = []  # stores all the possible outcomes

        # Stochastic Action Model
        if dirs == "UP":
            move = ["UP","LEFT","RIGHT"]
        elif dirs == "DOWN":
            move = ["DOWN","RIGHT","LEFT"]
        elif dirs == "LEFT":
            move = ["LEFT","UP","DOWN"]
        else:
            move = ["RIGHT","UP","DOWN"]

        # effect of probabilities on the direction the agent moves
        probabilities = [0.8,0.1,0.1]

        # computes next state with direction and probability
        for a,b in zip(move,probabilities):
            upd_state = self.directions(state,a)
            
            # rewards given according to the event
            if upd_state == self.end:
                rwd = 10 
            else:
                rwd = -1 

            # stores transition
            transition_dir.append((upd_state,b,rwd))
        return transition_dir

# TASK 1  
# VALUE ITERATION - Computes optimal value function using Bellman optimality equation 

def value_iter(grid):

    value = {} # value function initialised 
    # all values in the grid are initialised 
    for i in range(grid.grid_size):         
        for j in range(grid.grid_size):
            value[(i,j)] = 0 
    # convergence threshold 
    stopping_val = 0.0001

    while True:
        delta = 0   # records the maximum change 
        new_value = value.copy()        # prevents updating the old dictionary 

        for s in value:    # skips the goal square and the blockers square 
            if s in grid.blockers or s == grid.end:
                continue
            agent_action_val = []           # stores Q-values 
            for a in grid.agent_actions:    # loops through actions that can be taken by the agent
                val = 0 
                for upd_state,p,r in grid.transition_model(s,a):
                    val += p*(r+grid.gamma_val*value[upd_state])       # Computes Q(s,a)

                agent_action_val.append(val)

            best_action = max(agent_action_val)     # V(s) = max Q(s,a)
            new_value[s] = best_action              # records updated value
            delta = max (delta, abs(best_action-value[s])) 
        value = new_value   # updates value table 
        if delta<stopping_val:  # check if converges
            break 
    policy_dict = {}    # extracts policy
    for s in value:
        if s in grid.blockers or s == grid.end:
            continue        # invalid states are skipped 
        best_action = None 
        best_value = float("-inf")         
        for a in grid.agent_actions:
            val = 0     # action value initialised for each action 
            for upd_state,p,r in grid.transition_model(s,a):    
                    val += p*(r+grid.gamma_val*value[upd_state])    # Bellman Expectation Calculation 
            if val > best_value:     # Q-value of the action is compared with the best action found so far
                best_value = val
                best_action = a         
        
        policy_dict[s] = best_action    # Optimal action is stored 
    return value,policy_dict

print("\nVALUE FUNCTION")

# POLICY ITERATION - Start from an initial policy and alternate between policy evaluation and policy improvement until convergence

def policy_iter(grid):

    policy_dict = {}
    for i in range(grid.grid_size):         
        for j in range(grid.grid_size):
            s = (i,j)
            if s in grid.blockers or s == grid.end:
                continue 
            policy_dict[s] = grid.agent_actions[0]

    value = {} # value function initialised 
    # all values in the grid are initialised 
    for i in range(grid.grid_size):         
        for j in range(grid.grid_size):
            value[(i,j)] = 0 

    while True:

        # POLICY EVALUATION 
        while True:
            delta = 0   # records the maximum change 
            new_value = value.copy()        # prevents updating the old dictionary 
            for s in value:    # skips the goal square and the blockers square 
                if s in grid.blockers or s == grid.end:
                    continue
                action = policy_dict[s]
                val = 0 
                for upd_state,p,r in grid.transition_model(s,action):
                        val += p*(r+grid.gamma_val*value[upd_state])       # Computes Q(s,a)

                new_value[s] = val              # records updated value
                delta = max (delta, abs(val-value[s])) 
            value = new_value   # updates value table 
            if delta<0.0001:  # check if converges
                break 

        # POLICY IMPROVEMENT
        policy_stable = True 
        for s in policy_dict:
            prev_action = policy_dict[s]
            best_action = None 
            best_value = float("-inf")         
            for a in grid.agent_actions:
                val = 0     # action value initialised for each action 
                for upd_state,p,r in grid.transition_model(s,a):    
                        val += p*(r+grid.gamma_val*value[upd_state])    # Bellman Expectation Calculation 
                if val > best_value:     # Q-value of the action is compared with the best action found so far
                    best_value = val
                    best_action = a         
            policy_dict[s] = best_action
            if best_action != prev_action:
                policy_stable = False
        if policy_stable:
            break   
    return value,policy_dict

## TASK 2 
# EPISODE SIMULATION 
def episode(grid, Q_sa,epsilon = 0.1):
    eps = []            # list of episodes - stored in the format: (state,action,reward)
    state = grid.start  # start state of every episode - (0,0)
    steps = 0 
    max_steps = 100 
   
    while state != grid.end and steps<max_steps:    # episode continues until goal is reached 
        if ran.random() < epsilon:
            action = ran.choice(grid.agent_actions)
        else:
            q_vals = [Q_sa[(state,a)] for a in grid.agent_actions]
            best_index = q_vals.index(max(q_vals))
            action = grid.agent_actions[best_index]
        transitions = grid.transition_model(state,action)  # returns possible outcomes [(next_state,probability,reward)]
        
        probabilities = [t[1] for t in transitions]     # extracts probabilities
        upd_state,p,rwd = ran.choices(transitions, weights = probabilities)[0]  # outcome chosen according to probabilities
        eps.append((state,action,rwd))    # adds to episode history 
        state = upd_state   
        
    return eps      # returns full trajectory

# MONTE CARLO CONTROL ALGORITHM - learns optimal policy using Monte Carlo Learning 
def monte_carlo_algorithm(grid,episodes = 10000):
    Q_sa = {} # stores the expected return when taking action a in state s 
    results = {} # stores all the returns and then average them 
    policy = {} # stores current policy 
    # loops through the grid
    for i in range(grid.grid_size):
        for j in range(grid.grid_size):
            s = (i,j) # represents the state 
            if not grid.state_check(s):
                continue 
            # Loops through the actions
            for a in grid.agent_actions:
                Q_sa[(s,a)] = 0
                results[(s,a)] = []
    # Learning done through the episodes 
    for _ in range(episodes):
        eps = episode(grid,Q_sa)
        G = 0 # calculates total future reward 
        visited = set() # keeps tract of the visited pairs 
        
        # ESTIMATING THE STATE ACTION VALUES  USING SAMPLED RETURNS 
        for s,a,r in reversed(eps): # calculates results from the end of the episode backward
            G = grid.gamma_val*G + r # sample returns
            if (s,a) not in visited:  # each pair updated once per episode 
                results[(s,a)].append(G)
                Q_sa[(s,a)] = np.mean(results[(s,a)])   
                best_action = max(grid.agent_actions,key = lambda a: Q_sa[(s,a)])    # selects action with highest Q-value
                # policy[s] = best_action  #policy improvement step 
                visited.add((s,a))

        policy = {}

    for i in range(grid.grid_size):
        for j in range(grid.grid_size):

            s = (i,j)

            if s in grid.blockers or s == grid.end:
                continue

            q_vals = [Q_sa[(s,a)] for a in grid.agent_actions]

            best_index = q_vals.index(max(q_vals))

            policy[s] = grid.agent_actions[best_index]

    return policy
    

# TASK 3
# Q Learning - function teaches the agent how to reach the goal in the grid by learning from experience 

def q_Learning(grid,episodes=10000,alpha = 0.1,epsilon = 0.1):
    # Tabular Q-Learning - stores expected value of taking action a in state s 
    Q_sa = {}
    policy = {}
    # loops through the grid
    for i in range(grid.grid_size):
        for j in range(grid.grid_size):
            s = (i,j) # represents the state 
            # Loops through the actions
            for a in grid.agent_actions:
                Q_sa[(s,a)] = 0
   # Learning done through the episodes 
    for _ in range(episodes):
        s = grid.start # agent begins at (0,0)
        steps = 0 
        while s!=grid.end and steps < 100: # this episode continues until the goal state is reached     
            steps +=1

            # E- greedy Action Selecion 
            if ran.random()<epsilon:
                action = ran.choice(grid.agent_actions)
            else:
                action = max(grid.agent_actions,key=lambda a: Q_sa[(s,a)])
            transitions = grid.transition_model(s,action)  # environment transition - returns possible outcomes [(next_state,probability,reward)]

            # agent samples one outcome 
            probabilities = [t[1] for t in transitions]     # extracts probabilities
            upd_state,p,rwd = ran.choices(transitions, weights = probabilities)[0]  # outcome chosen according to probabilities    
            max_next_Q = max(Q_sa[(upd_state,a)] for a in grid.agent_actions)   # calculates best future action value 

            # Q Learning update rule 
            Q_sa[(s,action)] += alpha * (rwd +grid.gamma_val *max_next_Q - Q_sa[(s,action)])
            s = upd_state # move to next state and continues episode 
    # After training, convert Q table into policy 
    for i in range(grid.grid_size):
        for j in range(grid.grid_size):
            s = (i,j) # represents the state 
            if s in grid.blockers or s == grid.end:
                continue 
            best = max(Q_sa[(s,a)] for a in grid.agent_actions) # Chooses the function with the highest Q-value 
            best_actions = [a for a in grid.agent_actions if Q_sa[(s,a)] == best]
            policy[s] = ran.choice(best_actions) 
    return policy   # returns optimal policy learned through  Q - learning 

# Part 2 Working 

def pt2_gridworld():
    env_gridworld = GridWorldenv()         # environment is created 
#TASK 1 
    print("VALUE ITERATION")
    value,policy = value_iter(env_gridworld)    # runs the value iteration
    for j in reversed(range(env_gridworld.grid_size)):
        for i in range(env_gridworld.grid_size):
            print(f"{value[(i,j)]:6.2f}", end=" ")
        print()
    for j in reversed(range(env_gridworld.grid_size)):    # Top row is printed 
        row_grid = ""       # empty row string 
        for i in range(env_gridworld.grid_size):         # left to right across th grid 
            state = (i,j)  # current state
            if state in env_gridworld.blockers:     # check for obstacles 
                row_grid += " X "
            elif state == env_gridworld.end:        # check for the terminal state
                row_grid += " G "
            
            elif state == env_gridworld.start:
                row_grid += " S "
            else:
                row_grid += " "+ policy.get(state,".")[:1] + " "    # return optimal action for the state
        print(row_grid)     # displays the row 
    
    print("\nPOLICY ITERATION")
    val_pi,priority_pi = policy_iter(env_gridworld)
    for j in reversed(range(env_gridworld.grid_size)):
        for i in range(env_gridworld.grid_size):
            print(f"{val_pi[(i,j)]:6.2f}", end=" ")
        print()

    for j in reversed(range(env_gridworld.grid_size)):    # Top row is printed 
        row_grid = ""       # empty row string 
        for i in range(env_gridworld.grid_size):         # left to right across th grid 
            state = (i,j)  # current state
            if state in env_gridworld.blockers:     # check for obstacles 
                row_grid += " X "
            elif state == env_gridworld.end:        # check for the terminal state
                row_grid += " G "
            
            elif state == env_gridworld.start:
                row_grid += " S "

            else:
                row_grid += " "+ priority_pi.get(state,".")[:1] + " "    # return optimal action for the state
        print(row_grid)     # displays the row 
# TASK 2 
    print("\nMONTE CARLO CONTROL")
    mc = monte_carlo_algorithm(env_gridworld,episodes = 10000)
    for j in reversed(range(env_gridworld.grid_size)):    # Top row is printed 
        row_grid = ""       # empty row string 
        for i in range(env_gridworld.grid_size):         # left to right across th grid 
            state = (i,j)  # current state
            if state in env_gridworld.blockers:     # check for obstacles 
                row_grid += " X "
            elif state == env_gridworld.end:        # check for the terminal state
                row_grid += " G "
            elif state == env_gridworld.start:
                row_grid += " S "
            else:
                row_grid += " "+ mc.get(state,".")[:1] + " "    # return optimal action for the state
        print(row_grid)     # displays the row 
# TASK 3 
    print("\nQ Learning")
    ql = q_Learning(env_gridworld,episodes=10000)
    for j in reversed(range(env_gridworld.grid_size)):    # Top row is printed 
        row_grid = ""       # empty row string 
        for i in range(env_gridworld.grid_size):         # left to right across th grid 
            state = (i,j)  # current state
            if state in env_gridworld.blockers:     # check for obstacles 
                row_grid += " X "
            elif state == env_gridworld.end:        # check for the terminal state
                row_grid += " G "
            elif state == env_gridworld.start:
                row_grid += " S "
            else:
                row_grid += " "+ ql.get(state,".")[:1] + " "    # return optimal action for the state
        print(row_grid)     # displays the row 

if __name__ == "__main__":
    pt2_gridworld()