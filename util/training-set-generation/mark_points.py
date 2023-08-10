import math
from turtle import *
import matplotlib.pyplot as plt
import random
from graph.graph import Node, Undirected_Graph
import numpy as np

def gridAll(graph : Undirected_Graph, list_polygon : list=[]):
        polygons = []
        while (not graph.if_all_nodes_marked()):
                node = graph.get_random_Not_Marked_Node()
                polygon = check_polygon(graph=graph, start=node)
                polygons.append(polygon)

        return polygons

def check_polygon(graph : Undirected_Graph, start : Node, path : list=[], prev : Node = None):
        path.append(start)
        grids = []
        start.marked = True
        neighbor_nodes = graph.get_neighbors(start)
        if (neighbor_nodes == None):
                return grids
        path_copy = path.copy()
        for neighbor in neighbor_nodes:
                graph.mark_edge(start, neighbor)
                if (neighbor == prev):
                        continue
                if (neighbor.marked == False):
                        if(len(grids)==0):
                                grids = check_polygon(graph=graph, start=neighbor, path=path_copy, prev=start)
                        else:
                                grids.append(check_polygon(graph=graph, start=neighbor, path=path_copy, prev=start))
                else:
                        count = 0
                        grid = []
                        for node in path:
                                if (neighbor.x == node.x and neighbor.y == node.y):
                                        grid = path [count:]
                                        '''if (grid.__len__()<30):
                                                grid = []
                                                break'''
                                        break
                                count += 1
                        grids.append(grid)

        return grids




def eliminate_edges(graph : Undirected_Graph):
        property_list = graph.properties.copy()
        for edge in property_list:
                node1 = graph.get_node(edge[0][0], edge[0][1])
                node2 = graph.get_node(edge[1][0], edge[1][1])
                property=None
                try:
                        property = graph.properties[(edge[0], edge[1])]
                except:
                        continue
                marked = property[1]

                if (marked == True):
                        neighborSet1 = []
                        neighborSet2 =[]

                        neighborSet1=graph.get_neighbors(node1)
                        neighborSet2=graph.get_neighbors(node2)
                        state=False
                        for node2 in neighborSet1:
                                marked = graph.get_marked(node1, node2)
                                if ( not marked):
                                        state = True
                                        break
                        if (state == False):
                                for node2 in neighborSet1:
                                        marked = graph.get_marked(node1, node2)
                                        if (not marked):
                                                state = True
                                                break
                        if (state):
                                graph.remove_edge(node1, node2)

def recursively_print(list : list):
        l=[]
        try:
                if(type((list[0]) != type(l))):
                        print("grid:")
        except:
                pass
        for item in list:
                if (type(item) == type(l) ):
                        recursively_print(item)
                else:
                        print(str(item))

def recursively_see(list : list):
        l=[]
        for item in list:
                if (type(item) == type(l) ):
                        recursively_print(item)
                else:
                        goto(item.x, item.y)

def recursively_add(list : list):
        l=[]
        result = []
        try:
                if(type((list[0]) != type(l))):
                        print("grid:")
        except:
                pass
        temp =[]
        for item in list:
                if (type(item) == type(l) ):
                        r = recursively_add(item)
                        if (r != None):
                                for grids in r:
                                        result.append(grids)
                else:
                        temp.append(item)
        result.append(temp)

        for item in result:
                if (type(item)==type(l) and len(item)==0):
                        result.remove(item)

        return result
def to_2DMAP(grids : list, graph : Undirected_Graph):
        arr2D = np.asarray(dtype = np.int32, a = np.zeros(shape = (512, 512)))
        graph.scale_Map(512)

def node_distance(node1 : Node, node2 : Node):
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
def mark_polygon(graph : Undirected_Graph, list_polygon : list=[]):
        pass


# no need to seperate polygon, since it will be seperated when marking masks

graph = Undirected_Graph()
nodes = []
edges = []
for i in range(0, 6):
        node = Node(random.randint(1, 20), random.randint(1, 20))
        graph.add_node(node)
        nodes.append(node)

for node in nodes:
        i=random.randint(0, len(nodes)-1)
        graph.add_edge(node, nodes[i])

        j=random.randint(0, len(nodes)-1)
        graph.add_edge(node, nodes[j])

        edges.append( (node.x, node.y, nodes[i].x, nodes[i].y) )
        edges.append( (node.x, node.y, nodes[j].x, nodes[j].y) )

'''
graph = Undirected_Graph()
nodes  =[(1,1), (2,2), (2,1), (1,2), (3,1), (3,2), (3,3), (2,3), (1,3)]
edges = [((1,1), (2,1)), ((1,1), (1,2)), ((2,1), (2,2)), ((1,2), (2,2)), ((2,3), (3,3))]

for node in nodes:
        node1= Node(node[0], node[1])
        graph.add_node(node1)
for edge in edges:
        n1 = graph.get_node(edge[0][0], edge[0][1])
        n2 = graph.get_node(edge[1][0], edge[1][1])
        graph.add_edge(n1, n2)

graph.add_edge(graph.get_node(1,1), graph.get_node(2,2))
        
grids = gridAll(graph)
recursively_print(grids)'''
#plot all the nodes and edges



grids = gridAll(graph)
points = []
recursively_print(grids)

g = recursively_add(grids)
recursively_print(g)


for node in nodes:
        points.append((node.x, node.y))

x1 = []
y1 = []
x2 = []
y2 = []

for edge in edges:
        x1.append(edge[0])
        y1.append(edge[1])
        x2.append(edge[2])
        y2.append(edge[3])

x, y = zip(*points)
plt.scatter(x, y)
plt.plot(x1, y1, x2, y2)

plt.show()






