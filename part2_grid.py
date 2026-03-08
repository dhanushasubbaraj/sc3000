import numpy as np 
import random as r 

# GRID WORLD 

# Environment set-up
class gridWorldenv:

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
    def dir(self,state,dir):
        x,y = state                # extracts coordinates

        # Movement of the agent 
        if dir == "UP":
            upd_state = (x,y+1)

        if dir == "DOWN":
            upd_state = (x,y-1)

        if dir == "RIGHT":
            upd_state = (x-1,y)

        if dir == "LEFT":
            upd_state = (x+1,y)   

        # checks if movements hits roadblockers - agent stays in the same state 
        if not self.state_check(upd_state):
            return state 
        
        return upd_state


    # TRANSITION MODEL - computes the probability of reaching next state from the current state after taking action 
    def transition_model(self,state,dir):
        if state == self.end:
            return [(state,1,0)]
        transition_dir = [] 
        if dir == "UP":
            move = ["UP","LEFT","RIGHT"]
        elif dir == "DOWN":
            move = ["DOWN","LEFT","RIGHT"]
        elif dir == "LEFT":
            move = ["LEFT","UP","DOWN"]
        else:
            move = ["UP","LEFT","RIGHT"]

        prob = [0.8,0.1,0.1]









# Part 2 Working 

def pt2_gridworld():
    env_gridworld = pt2_gridworld()
    print("VALUE ITERATION")
