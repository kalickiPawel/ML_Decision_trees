class Node:
    def __init__(self, parent, depth, gid):
        self.parent = parent
        self.depth = depth
        self.gid = gid
        self.left = None
        self.right = None
        self.feature_index = None
        self.pi = None
        self.yi = None

    def __str__(self):
        return f'Node: {self.gid}'
