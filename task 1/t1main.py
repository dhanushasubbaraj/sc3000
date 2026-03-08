import json
import task1


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

