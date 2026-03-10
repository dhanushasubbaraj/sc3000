import heapq


def UCS(src, goal, G, Dist, Cost):

    #initialise priority queue
    pq = []
    heapq.heappush(pq, (0, src))

    best_distance = {src: 0} # stores shortest known distance from src to each node
    parent = {} # stores previous node used to reach each node

    while pq:

        distance, node = heapq.heappop(pq) # pop lowest distance node first

        if node == goal: # optimal shortest path found
            break

        for neighbor in G[node]: # iterate through all neighbours

            edge_key = node + "," + neighbor
            new_dist = distance + Dist[edge_key]

            if neighbor not in best_distance or new_dist < best_distance[neighbor]: # check if this path is better than any previous paths found

                best_distance[neighbor] = new_dist # update shortest known distance
                parent[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor)) # add neighbour into priority queue

    path = []
    n = goal

    while n != src: #trace path backwards
        path.append(n)
        n = parent[n]

    path.append(src)
    path.reverse()

    # compute total energy cost of path
    total_energy = 0
    for i in range(len(path)-1):
        total_energy += Cost[path[i] + "," + path[i+1]]

    print("Shortest path:", "->".join(path))
    print("Shortest distance:", best_distance[goal])
    print("Total energy cost:", total_energy)