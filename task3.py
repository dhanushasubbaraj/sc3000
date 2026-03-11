import heapq
import math

ENERGY_BUDGET = 287932 # take energy constraints into consideration


def heuristic(node, goal, Coord):
    x1, y1 = Coord[node]
    x2, y2 = Coord[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) # use Euclidean distance to estimate how far a node is from the goal


def A_star(src, goal, G, Dist, Cost, Coord):

    pq = [] #initialise priority queue
    start_h = heuristic(src, goal, Coord)

    heapq.heappush(pq, (start_h, 0, 0, [src])) # (f, g, energy, path)

    visited = {} # node -> list of (distance, energy)

    while pq:

        f, distance, energy, path = heapq.heappop(pq)
        node = path[-1] # returns path with lowest f value 

        if node == goal:
            print("Shortest path:", "->".join(path))
            print("Shortest distance:", distance)
            print("Total energy cost:", energy)
            return

        for neighbor in G[node]: # iterate through all neighbours

            edge = node + "," + neighbor
            new_distance = distance + Dist[edge]
            new_energy = energy + Cost[edge]

            if new_energy > ENERGY_BUDGET: # check with energy constraints
                continue

            dominated = False # check and discard if another path to the same node already has lower distance or lower energy


            if neighbor in visited:
                for d, e in visited[neighbor]:
                    if new_distance >= d and new_energy >= e:
                        dominated = True
                        break

            if dominated:
                continue # to help reduce search space

            new_path = path + [neighbor]
            h = heuristic(neighbor, goal, Coord) # estimate how far neighbour is from goal
            new_f = new_distance + h

            heapq.heappush(pq, (new_f, new_distance, new_energy, new_path))

            if neighbor not in visited:
                visited[neighbor] = []

            visited[neighbor].append((new_distance, new_energy)) # record that node is visited

    print("No feasible path found")