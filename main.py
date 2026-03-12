import json
import task1
import task2
import task3
from part2_grid import pt2_gridworld

# PART 1

# open json files
g = open("G.json")
cost = open("Cost.json")
coord = open("Coord.json")
dist = open("Dist.json")

# load json files
G = json.load(g)
Cost = json.load(cost)
Coord = json.load(coord)
Dist = json.load(dist)

#print results of all tasks
print()
print("Part 1:")
print()

print("Task 1:")
print()
task1.UCS("1", "50", G, Dist, Cost)

print()
print("Task 2:")
print()
task2.UCS_energy("1", "50", G, Dist, Cost)

print()
print("Task 3:")
print()
task3.A_star("1", "50", G, Dist, Cost, Coord)

# PART 2 
print()
print ("Part 2: Solving MDP and Reinforcement Learning Problems Using a Grid World")
pt2_gridworld()     

