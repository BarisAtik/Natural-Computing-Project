import matplotlib.pyplot as plt
import csv

class Node:
    def __init__(self, x, y, adjacent_nodes=[]):
        self.x = x
        self.y = y
        self.adjacent_nodes = adjacent_nodes

class Edge:
    def __init__(self, source, target, speed_limit=60, pollution=0):
        self.source = source
        self.target = target
        self.speed_limit = speed_limit
        self.pollution = pollution


class Representation:
    def __init__(self, nodes_file, edges_file, maptype="sp"):

        # read the edges from the edges .csv file
        self.edges = []
        with open(edges_file) as f:
            reader = csv.reader(f)
            next(reader) # ignore header
            for line in reader:
                source, target = line
                source = int(source)
                target = int(target)
                self.edges.append(Edge(source, target))

        # read the nodes from the nodes file
        self.nodes = {}
        with open(nodes_file) as f:
            reader = csv.reader(f)
            next(reader) # ignore header
            for line in reader:
                id, x, y = line
                id = int(id)
                x = float(x)
                y = float(y)

                # find the adjacent nodes for each node
                adjacent_nodes = []
                for edge in self.edges:
                    if edge.source == id:
                        adjacent_nodes.append(edge.target)
                    if edge.target == id:
                        adjacent_nodes.append(edge.source)
                self.nodes[id] = Node(x, y, adjacent_nodes)

        # set the figure size and figure limits correctly
        self.figsize = (20,10) if maptype == "sp" else (5,8)
        self.figlims = (1232, 665) if maptype == "sp" else (850, 1000)

    def plot_map(self):
        plt.figure(figsize=self.figsize)
        for edge in self.edges:
            plt.plot([self.nodes[edge.source].x, self.nodes[edge.target].x], 
                     [self.figlims[1] - self.nodes[edge.source].y, self.figlims[1] - self.nodes[edge.target].y])
        plt.xlim(0,self.figlims[0])
        plt.ylim(0,self.figlims[1])
        plt.show()

    def plot_road(self):
        # TODO implement
        pass