import math
from copy import deepcopy
from collections import deque

class FibNode:
    def __init__(self, k, x):
        self.loser = False
        self.priority = k
        self.parent = set()
        self.siblings = set()
        self.children = set()
        self.real_world = x

    def decrease_priority(self, k):
        self.priority = k

class FibHeap:
    def __init__(self):
        self.roots = set()
        self.mapping = {}
        self.minroot = None

    def push(self, x, k):
        print(x, k)
        if x in self.mapping.keys():
            self.decreasekey(x, k)
        else:
            to_add = FibNode(k, x)
            self.mapping[x] = to_add
            self.roots.add(x)
            if self.minroot is None:
                self.minroot = x
            elif self.mapping[self.minroot].priority > k:
                self.minroot = x

    def decreasekey(self, x, k):
        if self.mapping[list(self.mapping[x].parent)[0]].priority > k:
            self.double_loser(x)
        self.mapping[x].decrease_priority(k)

    def popmin(self):
        # Remove minroot and promote children
        to_return = deepcopy(self.minroot)
        self.roots.remove(self.minroot)
        for child in self.mapping[self.minroot].children:
            self.roots.add(child)
        self.minroot = None

        # Cleanup
        dict_degrees = {}
        queue = deque()
        for root in self.roots:
            queue.append(root)
        self.roots = set()
        while len(queue) > 0:
            current = queue.pop()
            degree = len(self.mapping[current].children)
            if self.mapping[current].loser:
                degree += 1
            if degree in dict_degrees:
                if self.mapping[current].priority > self.mapping[dict_degrees[degree]].priority:
                    self.mapping[current].parent = {dict_degrees[degree]}
                    self.mapping[dict_degrees[degree]].children.add(current)
                    queue.append(dict_degrees[degree])
                    del dict_degrees[degree]
                else:
                    self.mapping[dict_degrees[degree]].parent = {current}
                    self.mapping[current].children.add(dict_degrees[degree])
                    queue.append(current)
                    del dict_degrees[degree]
            else:
                dict_degrees[degree] = current
        for key in dict_degrees.keys():
            self.roots.add(dict_degrees[key])

        # Find new minroot
        for root in self.roots:
            if self.minroot is None:
                self.minroot = root
            elif self.mapping[self.minroot].priority > self.mapping[root].priority:
                self.minroot = root

        return to_return


    def __contains__(self, x):
    # returns True if the heap contains item x, False otherwise
        for key in self.mapping.keys():
            if key==x:
                return True
        return False

    def __bool__(self):
        return len(self.roots) > 0
    # returns True if the heap has any items at all, False otherwise

    def double_loser(self, x):
        self.mapping[x].loser = False
        if x in self.roots:
            return
        parent = list(self.mapping[x].parent)[0]
        self.mapping[parent].children.remove(x)
        self.mapping[x].parent = set()
        self.roots.add(x)
        if self.mapping[parent].loser:
            self.double_loser(parent)
        else:
            self.mapping[parent].loser = True

