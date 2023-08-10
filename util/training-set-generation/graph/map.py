import graph
import numpy as np

class Map:
    def __init__(self, width, height, accuracy):
        self.width = width
        self.height = height
        self.accuracy = accuracy
        self.graph = graph.Undirected_Graph()
