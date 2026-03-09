import json
import task1
import task2
import task3

# Load JSON files
g = open("G.json")
cost = open("Cost.json")
coord = open("Coord.json")
dist = open("Dist.json")

G = json.load(g)
Cost = json.load(cost)
Coord = json.load(coord)
Dist = json.load(dist)

print("Task 1:")
print()
task1.UCS("1", "50", G, Dist, Cost)

print("Task 2:")
print()
task2.UCS_energy("1", "50", G, Dist, Cost)

print("Task 3:")
print()
task3.A_star("1", "50", G, Dist, Cost, Coord)
