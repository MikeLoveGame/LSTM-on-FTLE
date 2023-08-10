class Set:
    def __init__(self, ID : int):
        self.nodes = []
        self.group = ID
    def add(self, node):
        self.nodes.append(node)
    def check_group(self, set):
        for n in self.nodes:
            for m in set.nodes:
                if n.x == m.x and n.y == m.y:
                    return True
        return False