from graph.graph import Node, Undirected_Graph
from matplotlib import pyplot as plt
from mark_points import *
import sys
from turtle import *

sys.setrecursionlimit(1000000000)

def readNode(f, graph : Undirected_Graph):
    nodeList = []
    while(True):
        s = f.__next__()
        s1 = s.split(" ")

        count = 0
        x = 0
        y = 0

        if (s.__contains__("METADATA")):
            break

        for string in s1:
            if (string == '\n'):
                continue
            if (count == 0):
                x= round(float(string))
            elif(count == 1):
                y = round(float(string))
            else:
                node = Node(x, y)
                nodeList.append(node)
                count = -1
            count+=1


    x = []
    y = []
    for node in nodeList:
        x.append(node.x)
        y.append(node.y)

    count =0
    for node in nodeList:
        graph.add_node(node)
        print("adding node:" + str(count))
        count += 1

    plt.scatter(x, y)
    plt.show()
    return nodeList



def create_edge(f, nodeList : list, graph : Undirected_Graph):

    connectivity = -1
    count = 1

    while(True):
        s = f.__next__()
        if (s.__contains__("CELL")):
            break
        s1 = s.split(" ")
        for string in s1:

            if (string == '\n'):
                continue
            val = int(string)
            if (val == connectivity):
                graph.add_edge(nodeList[val], nodeList[val+count])
                print(f"adding edge: {nodeList[val]}, {nodeList[val+count]}")
                count += 1
            else:
                connectivity = val
                count=1

def edge_attempt(graph : Undirected_Graph):
    nodeList = graph.nodes
    for node1 in nodeList:
        for node2 in nodeList:
            if (node1 != node2  and node_distance(node1, node2) < 1.2):
                graph.add_edge(node1, node2)
                print(f"adding edge: {node1}, {node2}")

def check_Conncetivity(f):
    connectivity = 0
    count = 1

    while (True):
        s = f.__next__()
        if (s.__contains__("CELL")):
            break
        s1 = s.split(" ")
        for string in s1:
            if (string == '\n'):
                continue
            val = int(string)
            if (val == connectivity):
                count +=1
            else:
                if(count < 2):
                    print(f"connectivity: {connectivity} is less than 2 ")
                connectivity = val
                count = 1
def create_Graph(file_path : str):

    HEADER = "# vtk DataFile Version 5.1 \n vtk output \n ASCII"



    graph = Undirected_Graph()


    f = open(file_path, "r")
    i = 0
    nodeList = []

    for line in f:
        if (line.__contains__("POINT") and  not line.__contains__("DATA")):
            nodeList = readNode(f, graph=graph)
        '''elif(line.__contains__("CONNECTIVITY")):
            create_edge(f, nodeList=nodeList, graph=graph)'''

    f.close()
    return graph

def node_reduce(graph : Undirected_Graph, critical_nodes : list) -> Undirected_Graph:
    critical_graph = Undirected_Graph()
    nodeList = []
    for node in critical_nodes:
        node = graph.get_node(node.x, node.y)
        if(node is None):
            print("node is none")
        nodeList.append(node)


def mark_polygon(polygons : list) -> list:
    #return a 2D marked sections let 0 be unmarked, otherwise marked
    pass

def see_grids(nodeList : list):
    begin_fill()
    for grid in nodeList:
        for node in grid:
            goto(node.x, node.y)

def see_grids2(nodeList : list):
    for grid in nodeList:
        x = []
        y = []
        for node in grid:
            x.append(node.x)
            y.append(node.y)
        plt.scatter(x, y)
    plt.show()
def main():
    graph = create_Graph(file_path=r"C:\AI\Data\PV-state\pipline-operation\sample.vtk")
    edge_attempt(graph=graph)
    polygons = []
    while(not graph.if_all_nodes_marked()):
        node = graph.get_random_Not_Marked_Node()
        polygon = check_polygon(graph=graph, start=node)
        polygons.append(polygon)

    grids = recursively_add(polygons)

main()