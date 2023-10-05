import math
from copy import deepcopy
import random


def swap(arr, n, m):
    temp = arr[n]
    arr[n] = arr[m]
    arr[m] = temp


class KeyValuePair:
    def __init__(self, key, value, exists):
        self.key = key
        self.value = value
        self.exists = exists


class Heap:
    def __init__(self, key_values):
        self.size = 0
        self.array = [KeyValuePair(None, None, False) for i in range(self.size + 1000)]
        for current in key_values:
            self.insert(current)

    def insert(self, key_value_pair):
        self.array[self.size] = key_value_pair
        index = self.size
        self.size += 1
        in_position = False
        while (not in_position) and (index != 0):
            if self.array[index].key < self.array[math.floor((index - 1) / 2)].key:
                swap(self.array, index, math.floor((index - 1) / 2))
                index = math.floor((index - 1) / 2)
            else:
                in_position = True

    def pop(self):
        to_return = self.array[0]
        self.remove_first()
        self.size -= 1
        return to_return

    def normalise(self):
        if self.size * 2 > len(self.array):
            self.array += [KeyValuePair(None, None, False) for i in range(self.size)]

    def remove_first(self, index=0):
        left = 2 * index + 1
        right = 2 * index + 2
        if (not self.array[left].exists) and (not self.array[right].exists):
            self.delete(index)
            if self.array[index + 1].exists:
                swap(self.array, index, index + 1)
            return
        if (not self.array[right].exists) and self.array[left].exists:
            swap(self.array, left, index)
            return self.remove_first(left)
        else:
            assert self.array[left].exists
            assert self.array[right].exists
            if self.array[left].key < self.array[right].key:
                swap(self.array, index, left)
                return self.remove_first(left)
            else:
                swap(self.array, index, right)
                return self.remove_first(right)

    def delete(self, index):
        self.array[index] = KeyValuePair(None, None, False)

    def contains(self, key):
        for i in range(self.size):
            if self.array[i].key == key:
                return True
        return False

    def contains_value(self, value):
        for i in range(self.size):
            if self.array[i].value == value:
                return True
        return False

    def reduce_key(self, old_key, new_key):
        not_found = True
        count = -1
        while not_found:
            count += 1
            if self.array[count].key == old_key:
                not_found = False
                self.array[count].key = new_key
        index = count
        in_position = False
        while (not in_position) and (index != 0):
            if self.array[index].key < self.array[math.floor((index - 1) / 2)].key:
                swap(self.array, index, math.floor((index - 1) / 2))
                index = math.floor((index - 1) / 2)
            else:
                in_position = True

    def assertions(self):
        for i in range(len(self.array) - 1):
            assert not ((not self.array[i].exists) and self.array[i + 1].exists)

    def reduce_key_with_second_value(self, value, new_key):
        not_found = True
        count = -1
        while not_found:
            count += 1
            if self.array[count].value[1] == value:
                not_found = False
                self.array[count].key = new_key
        index = count
        in_position = False
        while (not in_position) and (index != 0):
            if self.array[index].key < self.array[math.floor((index - 1) / 2)].key:
                swap(self.array, index, math.floor((index - 1) / 2))
                index = math.floor((index - 1) / 2)
            else:
                in_position = True

    def reduce_key_with_value(self, value, new_key):
        not_found = True
        count = -1
        while not_found:
            count += 1
            if self.array[count].value == value:
                not_found = False
                self.array[count].key = new_key
        index = count
        in_position = False
        while (not in_position) and (index != 0):
            if self.array[index].key < self.array[math.floor((index - 1) / 2)].key:
                swap(self.array, index, math.floor((index - 1) / 2))
                index = math.floor((index - 1) / 2)
            else:
                in_position = True


class PriorityQueue:
    def __init__(self, key_values):
        self.queue = Heap(key_values)

    def pop(self):
        return self.queue.pop()

    def push(self, key_value_pair):
        self.queue.insert(key_value_pair)
        return

    def contains(self, key):
        return self.queue.contains(key)

    def contains_value(self, value):
        return self.queue.contains_value(value)

    def reduce_key_with_key(self, old_key, new_key):
        self.queue.reduce_key(old_key, new_key)

    def reduce_key_with_value(self, value, new_key):
        self.queue.reduce_key_with_value(value, new_key)

    def reduce_key_with_second_value(self, value, new_key):
        self.queue.reduce_key_with_second_value(value, new_key)

    def __len__(self):
        return self.queue.size


class Stack:
    def __init__(self):
        self.stack = []

    def pop(self):
        if len(self.stack) > 0:
            to_return = self.stack[0]
            self.stack = self.stack[1:]
            return to_return
        else:
            raise Exception("Stack is empty")

    def push(self, to_push):
        self.stack = [to_push] + self.stack
        return

    def push_end(self, to_push):
        self.stack += [to_push]
        return

    def __len__(self):
        return len(self.stack)


class Node:
    def __init__(self, value, edges):
        self.value = value
        self.edges = edges


class NodeWithReal(Node):
    def __init__(self, value, real, edges):
        super().__init__(value, edges)
        self.real = real


class CostedEdge:
    def __init__(self, source, cost, dest):
        self.source = source
        self.cost = cost
        self.dest = dest

    def add_to_cost(self, to_add):
        self.cost += to_add


class UndirectedCostedEdge:
    def __init__(self, node_one, node_two, cost):
        self.connects = {node_one, node_two}
        self.cost = cost

    def add_to_cost(self, to_add):
        self.cost += to_add


class UndirectedGraph:
    def __init__(self, nodes):
        self.nodes = {node.value: node.edges for node in nodes}

    def prims(self):
        distances = {node: math.inf for node in self.nodes.keys()}
        in_tree = {node: False for node in self.nodes.keys()}
        come_from = {node: None for node in self.nodes.keys()}
        s = next(iter(distances.keys()))
        distances[s] = 0  # The distance to the randomly chosen first node is 0
        to_explore = PriorityQueue([KeyValuePair(distances[s], s, True)])
        while not len(to_explore) == 0:
            v = to_explore.pop()
            in_tree[v.value] = True
            for edge in self.nodes[v.value]:
                w = self.find_other_connection(v.value, edge)
                if (not in_tree[w]) and edge.cost < distances[w]:
                    distances[w] = edge.cost
                    come_from[w] = v.value
                    if to_explore.contains_value(w):
                        to_explore.reduce_key_with_value(w, distances[w])
                    else:
                        to_explore.push(KeyValuePair(distances[w], w, True))
        edges = set(
            (node, come_from[node]) for node in self.nodes.keys())  # this includes (s, None) - which can be removed
        return edges

    def find_longest_path(self, node):
        stack = Stack()
        stack.push_end(node)
        visited = set()
        visited.add(node)
        distance = 0
        while len(stack) > 0:
            found_new_connection = False
            current = stack.pop()
            for edge in self.nodes[current]:
                if self.find_other_connection(current, edge) not in visited:
                    if not found_new_connection:
                        distance += 1
                        found_new_connection = True
                    stack.push_end(self.find_other_connection(current, edge))
                    visited.add(self.find_other_connection(current, edge))
        return distance

    def two_bfs_diameter(self):
        starting = random.choice(list(self.nodes.keys()))
        first_bfs_distance = self.bfs(starting)
        furthest_node = max(first_bfs_distance)
        second_bfs_distance = self.bfs(furthest_node)
        diameter = max(second_bfs_distance.values())
        return diameter

    def bfs(self, node):
        stack = Stack()
        stack.push_end(node)
        distances = {n: math.inf for n in self.nodes.keys()}
        distances[node] = 0
        while len(stack) > 0:
            current = stack.pop()
            for edge in self.nodes[current]:
                dest = self.find_other_connection(current, edge)
                if distances[dest] == math.inf:
                    stack.push_end(dest)
                    distances[dest] = distances[current] + 1
        return distances

    @staticmethod
    def find_other_connection(node, edge):
        for n in edge.connects:
            if n != node:
                return n

    def add_node(self, node):
        self.nodes[node.value] = node.edges

    def add_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.value] = node.edges


class DirectedGraph:
    def __init__(self, nodes):
        self.nodes = {node.value: node.edges for node in nodes}

    def add_node(self, node):
        self.nodes[node.value] = node.edges

    def add_nodes(self, nodes):
        for node in nodes:
            self.nodes[node.value] = node.edges

    def find_minimum_cost(self, start):  # Function to find the shortest path length using Dijkstra's algorithm
        costs = {node: math.inf if node != start else 0 for node in self.nodes.keys()}  # ASSERT no negative costs
        queue = PriorityQueue([KeyValuePair(0, start, True)])
        while len(queue) > 0:
            current = queue.pop()
            for edge in self.nodes[current.value]:
                dist_w = current.key + edge.cost
                if dist_w < costs[edge.dest]:
                    queue.push(KeyValuePair(dist_w, edge.dest, True))
                    costs[edge.dest] = dist_w
        return costs

    def find_minimum_reachable(self):
        queue = PriorityQueue({KeyValuePair(node, (
            [source_node for source_node in self.nodes if node in [edge.dest for edge in self.nodes[source_node]]],
            node), True) for node in self.nodes})

        lowest = {node: math.inf for node in self.nodes}
        while len(queue) > 0:
            current_lowest = queue.pop()
            lowest[current_lowest.value[1]] = current_lowest.key
            for to_change in current_lowest.value[0]:
                if lowest[to_change] == math.inf:
                    queue.reduce_key_with_second_value(to_change, current_lowest.key)
        return lowest

    def augmented_bellman_ford(self, start):  # Returns the negative weight cycle rather than reporting it exists
        lowest = {node: math.inf if node != start else 0 for node in self.nodes}
        edges = set(edge for node in self.nodes for edge in self.nodes[node])
        current_lowest = deepcopy(lowest)
        for i in range(len(lowest) - 1):
            for edge in edges:
                lowest[edge.dest] = min(current_lowest[edge.dest], current_lowest[edge.source] + edge.cost)
        current_lowest = deepcopy(lowest)
        for edge in edges:
            current_lowest[edge.dest] = min(lowest[edge.dest], lowest[edge.source] + edge.cost)
        all_pred = []
        if current_lowest != lowest:
            for i in range(len(lowest)):
                pred = []
                current_lowest = deepcopy(lowest)
                for edge in edges:
                    if lowest[edge.dest] > current_lowest[edge.source] + edge.cost:
                        pred.append((edge.source,
                                     edge.dest))  # This is part of a cycle - or by tracing backwards can lead to part of a cycle
                        lowest[edge.dest] = min(current_lowest[edge.dest], current_lowest[edge.source] + edge.cost)
                all_pred.append(pred)
        path = [all_pred[0][0][0], all_pred[0][0][1]]
        starting_node = all_pred[0][0][1]
        for i in range(1, len(lowest)):
            new_starting_node = 0
            for j in range(len(all_pred[i])):
                if starting_node == all_pred[i][j][0]:
                    if all_pred[i][j][1] in path:
                        return path
                    else:
                        new_starting_node = all_pred[i][j][1]
            starting_node = new_starting_node
            path.append(starting_node)
        return lowest

    def johnson(self):
        augmented_graph = deepcopy(self)  # Create a copy (by value) of the self to use as the helper graph
        augmented_graph.add_node(Node(0, {CostedEdge(0, 0, i) for i in self.nodes.keys()}))
        distances = augmented_graph.augmented_bellman_ford(0)  # ASSERT no negative weight cycles
        for node in augmented_graph.nodes.keys():
            for edge in augmented_graph.nodes[node]:
                edge.add_to_cost(distances[edge.source] - distances[edge.dest])
        del augmented_graph.nodes[0]
        costs_matrix = {}
        for node in set(augmented_graph.nodes.keys()):  # The node inserted to build the helper graph must now
            # be taken out
            min_costs = augmented_graph.find_minimum_cost(node)
            for item in min_costs.keys():
                costs_matrix[(node, item)] = min_costs[item] + distances[item] - distances[node]
        return costs_matrix


test = DirectedGraph({
    Node(1, {CostedEdge(1, 0, 3)}),
    Node(2, {CostedEdge(2, 0, 2)}),
    Node(3, {CostedEdge(3, 0, 1)})
})

a = test.find_minimum_reachable()
print(a)
