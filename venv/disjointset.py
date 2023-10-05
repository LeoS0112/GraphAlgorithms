from copy import deepcopy
from collections import deque


class DisjointSet:

    def __init__(self):
        self.sets = {}
        self.handles = {}

    def add(self, k):
        self.sets[k] = [k, []]
        self.handles[k] = k

    def __getitem__(self, k):
        current = deepcopy(k)
        while current != self.sets[current][0]:
            current = self.sets[current][0]
        return current

    def merge(self, h, i):
        self.sets[h][0] = i
        self.sets[i][1].append(h)
        del self.handles[h]

    def iter_from(self, k):
        to_return = set()
        queue = deque()
        queue.append(k)
        while len(queue) > 0:
            current = queue.pop()
            if current not in to_return:
                to_return.add(current)
                queue.append(self.sets[current][0])
                for to_add in self.sets[current][1]:
                    queue.append(to_add)
        return to_return


# Returns all items in the subset containing k

