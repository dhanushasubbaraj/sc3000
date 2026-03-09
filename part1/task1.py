import heapq


def UCS(src, goal, G, Dist, Cost):

    pq = []
    heapq.heappush(pq, (0, src))

    best_distance = {src: 0}
    parent = {}

    while pq:

        distance, node = heapq.heappop(pq)

        if node == goal:
            break

        for neighbor in G[node]:

            edge_key = node + "," + neighbor
            new_dist = distance + Dist[edge_key]

            if neighbor not in best_distance or new_dist < best_distance[neighbor]:

                best_distance[neighbor] = new_dist
                parent[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))

    path = []
    n = goal

    while n != src:
        path.append(n)
        n = parent[n]

    path.append(src)
    path.reverse()

    total_energy = 0
    for i in range(len(path)-1):
        total_energy += Cost[path[i] + "," + path[i+1]]

    print("Shortest path:", "->".join(path))
    print("Shortest distance:", best_distance[goal])
    print("Total energy cost:", total_energy)