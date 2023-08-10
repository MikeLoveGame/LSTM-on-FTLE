import numpy as np
import math

class Node:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.connected_edge = {}
        self.marked = False


    def __eq__(self, other):
        if isinstance(other, Node):
            return self.x == other.x and self.y == other.y
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def __str__(self):
        return str(self.x) + " " + str(self.y)


class Undirected_Graph:
    # nodes stored in [node1, node2, ...]
    # edge stored form in node1 : [node2, node3, ...]
    # properties stored in (node1, node2) : (distance, marked)
    def __init__(self):
        self.nodes = []
        self.edges = {}
        self.properties = {}
        self.max_x = 0
        self.max_y = 0
        self.min_x = 0
        self.min_y = 0

    def __deepcopy__(self, memodict={}):  # not exactly a deep copy, but it works
        new_map = Undirected_Graph()

        new_map.nodes = []
        for node in self.nodes:
            newNode = Node(node.x, node.y)
            new_map.nodes.append(newNode)
        for edge in self.properties:
            node1 = edge[0]
            node2 = edge[1]
            new_map.add_edge(node1, node2)
        return new_map

    def add_node(self, node: Node):
        x = node.x
        y = node.y
        if ( self.get_node(x, y) != None):
            return -1
        else:
            self.nodes.append(node)
            if (node.x > self.max_x):
                self.max_x = node.x
            if (node.y > self.max_y):
                self.max_y = node.y
            if (node.x < self.min_x):
                self.min_x = node.x
            if (node.y < self.min_y):
                self.min_y = node.y
            return 0
    def remove_node(self, node: Node):
        x = node.x
        y = node.y
        if ( self.get_node(x, y) == None):
            return -1
        else:
            self.nodes.remove(node)
            self.edges.pop(node)
            for key in self.edges:
                self.edges[key].remove(node)

            for node2 in self.nodes:
                try:
                    self.properties.pop((node, node2))
                except:
                    pass
                try:
                    self.properties.pop((node2, node))
                except:
                    pass
            return 0

    def get_node(self, x, y) -> Node:
        for node in self.nodes:
            if node.x == x and node.y == y:
                return node
        return None

    def get_node_index(self, x, y):
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.x == x and node.y == y:
                return i
        return -1

    def edge_exists(self, node1: Node, node2: Node) -> bool:
        if (node1 not in self.nodes or node2 not in self.nodes):
            return False
        try:
            self.properties[node1, node2]
            return True
        except:
            pass
        try:
            self.properties[node2, node1]
            return True
        except:
            return False

    def add_edge(self, node1: Node, node2: Node):
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y

        weight = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        if (x1 == x2 and y1 == y2):
            return -1

        if self.edge_exists(node1, node2):
            return -2
        else:
            if node1 not in self.edges:
                self.edges[node1] = [node2]
            else:
                self.edges[node1].append(node2)
            if node2 not in self.edges:
                self.edges[node2] = [node1]
            else:
                self.edges[node2].append(node1)

            if (node1, node2) in self.properties:
                self.properties[(node1, node2)] = (weight, False)
            else:
                self.properties[(node2, node1)] = (weight, False)
            return 0

    def mark_edge(self, node1: Node, node2: Node):
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y
        edges = self.edges[node1]
        for edge in edges:
            if edge.x == x2 and edge.y == y2:
                if (node1, node2) in self.properties:
                    self.properties[(node1, node2)] = (self.properties[(node1, node2)][0], True)
                else:
                    self.properties[(node2, node1)] = (self.properties[(node2, node1)][0], True)

                return 0
                break
        return -1

    def remove_edge(self, node1: Node, node2: Node):
        if (self.edge_exists(node1, node2) == False):
            return -1
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y
        edges = self.edges[node1]
        for edge in edges:
            if edge.__eq__(node2):
                self.edges[node1].remove(node2)
                self.edges[node2].remove(node1)
                if (node1, node2) in self.properties:
                    self.properties.pop((node1, node2))
                else:
                    self.properties.pop((node2, node1))

                return 0
                break

        return -1

    def get_weight(self, node1: Node, node2: Node):
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y
        edges = self.edges[node1]
        for edge in edges:
            if edge[0] == x2 and edge[1] == y2:
                if (node1, node2) in self.properties:
                    return self.properties[(node1, node2)][0]
                else:
                    return self.properties[(node2, node1)][0]
                break
        return -1

    def get_marked(self, node1: Node, node2: Node):
        x1 = node1.x
        y1 = node1.y
        x2 = node2.x
        y2 = node2.y
        edges = self.edges[node1]
        for edge in edges:
            if edge[0] == x2 and edge[1] == y2:
                if (node1, node2) in self.properties:
                    return self.properties[(node1, node2)][1]
                else:
                    return self.properties[(node2, node1)][1]
                break
        return None;

    def get_random_Not_Marked_Node(self):
        for node in self.nodes:
            if node.marked == False:
                return node
        return None
    def get_neighbors(self, node: Node):
        if node in self.edges:
            neighbors = self.edges[node]
            neighbor_nodes = []
            for neighbor in neighbors:
                neighbor_nodes.append(self.get_node(neighbor.x, neighbor.y))
            return neighbor_nodes
        else:
            return None

    def if_all_nodes_marked(self):
        for node in self.nodes:
            if node.marked == False:
                return False
        return True
    def scale_Map(self, factor):
        for node in self.nodes:
            ratio_x = node.x / self.max_x
            ratio_y = node.y / self.max_y
            node.x = ratio_x * factor
            node.y = ratio_y * factor
        self.max_x = factor
        self.max_y = factor

        for property in self.properties:
            weight = node_distance(property[0], property[1])
            self.properties[property] = (weight, self.properties[property][1])


def node_distance(node1: Node, node2: Node):
    return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

