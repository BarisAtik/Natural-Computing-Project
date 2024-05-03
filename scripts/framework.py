import matplotlib.pyplot as plt
import csv


class Node:
    def __init__(self, id, x, y, traffic, pollution, adjacent_nodes=[]):
        self.id = id
        self.x = x
        self.y = y
        self.coordinates = (x, y)
        self.traffic = traffic
        self.pollution = pollution
        self.adjacent_nodes = adjacent_nodes

    def __str__(self):
        return f"Node {self.id} at ({self.x}, {self.y})"


class Edge:
    def __init__(self, source, target, speed_limit=60, pollution=0):
        self.source = source
        self.target = target
        self.speed_limit = speed_limit
        self.pollution = pollution

    def __str__(self):
        return f"Edge from node {self.source} to node {self.target}"


class Representation:
    def __init__(self, nodes_file, edges_file, maptype="sp"):

        # read the edges from the edges .csv file
        self.edges = []
        with open(edges_file) as f:
            reader = csv.reader(f)
            next(reader)  # ignore header
            for line in reader:
                source, target = line
                source = int(source)
                target = int(target)
                self.edges.append(Edge(source, target))

        # read the nodes from the nodes file
        self.nodes = {}
        with open(nodes_file) as f:
            reader = csv.reader(f)
            next(reader)  # ignore header
            for line in reader:
                number, x, y, traffic, pollution = line
                number = int(number)
                x = float(x)
                y = float(y)
                traffic = float(traffic)
                pollution = float(pollution)

                # find the adjacent nodes for each node
                adjacent_nodes = []
                for edge in self.edges:
                    if edge.source == number:
                        adjacent_nodes.append(edge.target)
                    if edge.target == number:
                        adjacent_nodes.append(edge.source)
                self.nodes[number] = Node(number, x, y, traffic, pollution, adjacent_nodes)

        # set the figure size and figure limits correctly
        self.figsize = (20, 10) if maptype == "sp" else (5, 8)
        self.figlims = (1232, 665) if maptype == "sp" else (850, 1000)

        # set the map name correctly
        self.map_name = "Singapore" if maptype == "sp" else "the Netherlands"

    def plot_map(self, route=None, plot_nodes=False):
        plt.figure(figsize=self.figsize)
        for edge in self.edges:
            if (
                route
                and edge.source in route
                and (
                    (
                        (index := route.index(edge.source)) + 1 < len(route)
                        and edge.target == route[index + 1]
                    )
                    or (
                        (index := route.index(edge.source)) - 1 >= 0
                        and edge.target == route[index - 1]
                    )
                )
            ):
                color = "red"
                linewidth = 3
            else:
                color = "black"
                linewidth = 1
            plt.plot(
                [self.nodes[edge.source].x, self.nodes[edge.target].x],
                [
                    self.figlims[1] - self.nodes[edge.source].y,
                    self.figlims[1] - self.nodes[edge.target].y,
                ],
                color=color,
                linewidth=linewidth,
            )
        if route:
            plt.scatter(
                self.nodes[route[0]].x,
                self.figlims[1] - self.nodes[route[0]].y,
                color="black",
                zorder=3,
                s=50,
            )
            plt.text(
                self.nodes[route[0]].x + 5,
                self.figlims[1] - self.nodes[route[0]].y + 5,
                "Start",
                fontsize=12,
            )
            plt.scatter(
                self.nodes[route[-1]].x,
                self.figlims[1] - self.nodes[route[-1]].y,
                color="black",
                zorder=3,
                s=50,
            )
            plt.text(
                self.nodes[route[-1]].x + 5,
                self.figlims[1] - self.nodes[route[-1]].y + 5,
                "End",
                fontsize=12,
            )
        if plot_nodes:
            for id, node in self.nodes.items():
                plt.scatter(node.x, self.figlims[1] - node.y, color="black", s=12)
                plt.text(
                    node.x + 0.4,
                    self.figlims[1] - node.y,
                    f"{id}",
                    fontsize=8,
                )
        plt.xlim(0, self.figlims[0])
        plt.ylim(0, self.figlims[1])
        plt.show()
