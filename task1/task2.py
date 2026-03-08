import heapq

ENERGY_BUDGET = 287932


def UCS_energy(src, goal, G, Dist, Cost):

    pq = []
    heapq.heappush(pq, (0, 0, [src]))
    # (distance, energy, path)

    visited = {}  
    # node -> list of (distance, energy)

    while pq:

        distance, energy, path = heapq.heappop(pq)
        node = path[-1]

        if node == goal:

            print("Shortest path:", "->".join(path))
            print("Shortest distance:", distance)
            print("Total energy cost:", energy)
            return

        for neighbor in G[node]:

            edge = node + "," + neighbor

            new_distance = distance + Dist[edge]
            new_energy = energy + Cost[edge]

            if new_energy > ENERGY_BUDGET:
                continue

            dominated = False

            if neighbor in visited:
                for d, e in visited[neighbor]:
                    if new_distance >= d and new_energy >= e:
                        dominated = True
                        break

            if dominated:
                continue

            new_path = path + [neighbor]

            heapq.heappush(pq, (new_distance, new_energy, new_path))

            if neighbor not in visited:
                visited[neighbor] = []

            visited[neighbor].append((new_distance, new_energy))

    print("No feasible path found")