import time
from collections import deque
import heapq

graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 2), ('E', 5)],
    'C': [('F', 3)],
    'D': [('G', 3)],
    'E': [('G', 1)],
    'F': [('G', 2)],
    'G': []
}

h = {
    'A': 7,
    'B': 5,
    'C': 4,
    'D': 3,
    'E': 1,
    'F': 2,
    'G': 0
}

start = 'A'
goal = 'G'


def bfs():
    queue = deque([(start, [start])])
    visited = set()
    nodes = 0

    while queue:
        node, path = queue.popleft()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for next_node, cost in graph[node]:
            queue.append((next_node, path + [next_node]))

    return None, nodes


def dfs():
    stack = [(start, [start])]
    visited = set()
    nodes = 0

    while stack:
        node, path = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, nodes

        for next_node, cost in reversed(graph[node]):
            stack.append((next_node, path + [next_node]))

    return None, nodes


def a_star():
    queue = [(h[start], 0, start, [start])]
    visited = set()
    nodes = 0

    while queue:
        f, cost, node, path = heapq.heappop(queue)

        if node in visited:
            continue

        visited.add(node)
        nodes += 1

        if node == goal:
            return path, cost, nodes

        for next_node, edge_cost in graph[node]:
            new_cost = cost + edge_cost
            new_f = new_cost + h[next_node]
            heapq.heappush(
                queue,
                (new_f, new_cost, next_node, path + [next_node])
            )

    return None, 0, nodes


def hill_climbing():
    node = start
    path = [node]
    nodes = 1

    while node != goal:
        if not graph[node]:
            return None, nodes

        next_node = min(
            graph[node],
            key=lambda x: h[x[0]]
        )[0]

        if h[next_node] >= h[node]:
            return None, nodes

        node = next_node
        path.append(node)
        nodes += 1

    return path, nodes


colors = ['Red', 'Green', 'Blue']

csp = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}


def valid(node, color, assignment):
    for neighbour in csp[node]:
        if neighbour in assignment:
            if assignment[neighbour] == color:
                return False
    return True


def solve_csp(assignment):
    if len(assignment) == len(csp):
        return assignment

    for node in csp:
        if node not in assignment:
            break

    for color in colors:
        if valid(node, color, assignment):
            assignment[node] = color

            result = solve_csp(assignment)

            if result:
                return result

            del assignment[node]

    return None


print("SEARCH ALGORITHMS PERFORMANCE EVALUATION")
print("-" * 50)

t = time.perf_counter()
path, nodes = bfs()
bfs_time = time.perf_counter() - t

print("\nUninformed Search - BFS")
print("Path:", " -> ".join(path))
print("Nodes Explored:", nodes)
print("Time:", round(bfs_time, 6), "seconds")


t = time.perf_counter()
path, nodes = dfs()
dfs_time = time.perf_counter() - t

print("\nUninformed Search - DFS")
print("Path:", " -> ".join(path))
print("Nodes Explored:", nodes)
print("Time:", round(dfs_time, 6), "seconds")


t = time.perf_counter()
path, cost, nodes = a_star()
astar_time = time.perf_counter() - t

print("\nInformed Search - A*")
print("Path:", " -> ".join(path))
print("Path Cost:", cost)
print("Nodes Explored:", nodes)
print("Time:", round(astar_time, 6), "seconds")


t = time.perf_counter()
path, nodes = hill_climbing()
hill_time = time.perf_counter() - t

print("\nLocal Search - Hill Climbing")
print("Path:", " -> ".join(path))
print("Nodes Explored:", nodes)
print("Time:", round(hill_time, 6), "seconds")


t = time.perf_counter()
solution = solve_csp({})
csp_time = time.perf_counter() - t

print("\nConstraint Satisfaction - Backtracking")
print("Solution:", solution)
print("Time:", round(csp_time, 6), "seconds")

print("\n" + "-" * 50)
print("Evaluation Completed")