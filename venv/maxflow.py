import csv
from collections import deque
import math


class BFSVertex:
    def __init__(self, name, come_from, reachable):
        self.name = name
        self.come_from = come_from
        self.reachable_from_source = reachable


class FlowGraph:
    def __init__(self, capacities):
        self.capacities = capacities
        self.flow = {edge: 0 for edge in capacities.keys()}

    def calculate_flow(self, t):
        to_return = 0
        for u, v in self.flow.keys():
            if v == t:
                to_return += self.flow[(u, v)]
        return to_return

    def compute_augmenting_graph(self):
        augmenting_graph = {}
        for u, v in self.flow.keys():
            if self.flow[(u, v)] < self.capacities[(u, v)]:
                augmenting_graph[(u, v)] = 'inc'
            if self.flow[(u, v)] > 0:
                augmenting_graph[(v, u)] = 'dec'
        return augmenting_graph

    def find_augmenting_path(self, s, t):
        augmenting_graph = self.compute_augmenting_graph()
        queue = deque([s])
        visited_names = set()
        visited_names.add(s)
        visited = {s: BFSVertex(s, None, True)}
        not_found = True
        while (len(queue) > 0) and not_found:
            current = queue.popleft()
            for v, u in augmenting_graph.keys():
                if (v == current) and (u not in visited_names):
                    visited_names.add(u)
                    if v == s:
                        visited[u] = BFSVertex(u, v, True)
                    else:
                        visited[u] = BFSVertex(u, v, True)
                    queue.append(u)
                    if u == t:
                        not_found = False
        if not_found:
            return True, set(u for u in visited.keys() if visited[u].reachable_from_source)

        to_return = []
        current = t
        to_return.append(visited[current].name)
        while current != s:
            current = visited[current].come_from
            to_return.append(visited[current].name)

        to_return.reverse()
        return False, to_return

    def augment_flows(self, s, t):
        b, path = self.find_augmenting_path(s, t)
        if b:
            return True, path
        augmenting_graph = self.compute_augmenting_graph()
        delta = math.inf
        print(path)
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if augmenting_graph[edge] == 'inc':
                delta = min(delta, self.capacities[edge] - self.flow[edge])
            elif augmenting_graph[edge] == 'dec':
                delta = min(delta, self.flow[(path[i+1], path[i])])
            else:
                raise Exception
        for i in range(len(path) - 1):
            edge = (path[i], path[i+1])
            if augmenting_graph[edge] == 'inc':
                self.flow[edge] += delta
            elif augmenting_graph[edge] == 'dec':
                self.flow[(path[i+1], path[i])] -= delta
            else:
                raise Exception
        return False, None

    def find_max_flow(self, s, t):
        finished = False
        while not finished:
            b, cut = self.augment_flows(s, t)
            finished = b
        print(self.calculate_flow(t))
        print(self.flow)
        print(cut)
        return self.calculate_flow(t), self.flow, cut


def compute_max_flow(capacity, s, t):
    print(capacity)
    fg = FlowGraph(capacity)
    return fg.find_max_flow(s, t)


xa = 0.6 * 0.4 * 0.73 * 0.15
print(xa)
xb = 0.6 * 0.4 * 0.63 * 0.20
print(xb)
xaO = xa * 0.89
print(xaO)
xbO = xb * 0.1 * 0.75
print(xbO)








