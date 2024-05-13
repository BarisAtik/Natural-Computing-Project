import matplotlib.pyplot as plt
import csv


class Node:
    """
    Represents a node in the network.

    Attributes:
    - id: The unique identifier of the node.
    - coords: The coordinates of the node.
    - traffic: The traffic level at the node.
    - pollution: The pollution level at the node.
    - hotspots: The hotspot level at the node.
    - adjacent_nodes: The list of nodes adjacent to this node (as IDs).
    """

    def __init__(self, id, coords, traffic, pollution, hotspots, adjacent_nodes):
        self.id = id
        self.coords = coords
        self.traffic = traffic
        self.pollution = pollution
        self.hotspots = hotspots
        self.adjacent_nodes = adjacent_nodes

    def __str__(self):
        return f"Node {self.id} at ({self.coords[0]}, {self.coords[1]})"


class Edge:
    """
    Represents an edge in the network.

    Attributes:
    - source: The source node (ID) of the edge.
    - target: The target node (ID) of the edge.
    """

    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __str__(self):
        return f"Edge from node {self.source} to node {self.target}"


class Representation:
    """
    Represents the network as a graph.

    Attributes:
    - nodes: A dictionary of Node objects, with the node ID as the key.
    - edges: A list of Edge objects.
    - figsize: The size of the figure for plotting.
    - figlims: The limits of the figure for plotting.
    - scale_factor: The scale factor for plotting.
    - map_name: The name of the map.
    """

    def __init__(self, nodes_file, edges_file, maptype="sp"):
        self.edges = self.read_edges(edges_file)
        self.nodes = self.read_nodes(nodes_file, self.edges)
        self.figsize, self.figlims, self.scale_factor, self.map_name = (
            self.set_figure_properties(maptype)
        )

    def read_edges(self, edges_file):
        """
        Read the edges from a CSV file.

        Args:
        - edges_file: The path to the CSV file containing the edges.

        Returns:
        - A list of Edge objects.
        """

        edges = []
        with open(edges_file) as f:
            reader = csv.reader(f)
            next(reader)  # ignore header
            for line in reader:
                source, target = map(int, line)
                edges.append(Edge(source, target))
        return edges

    def read_nodes(self, nodes_file, edges):
        """
        Read the nodes from a CSV file.

        Args:
        - nodes_file: The path to the CSV file containing the nodes.
        - edges: A list of Edge objects.

        Returns:
        - A dictionary of Node objects, with the node ID as the key.
        """

        nodes = {}
        with open(nodes_file) as f:
            reader = csv.reader(f)
            next(reader)  # ignore header
            for line in reader:
                node_id, x, y, traffic, pollution, hotspots = map(float, line)
                node_id = int(node_id)
                adjacent_nodes = [
                    edge.target for edge in edges if edge.source == node_id
                ] + [edge.source for edge in edges if edge.target == node_id]
                nodes[node_id] = Node(
                    node_id, (x, y), traffic, pollution, hotspots, adjacent_nodes
                )
        return nodes

    def set_figure_properties(self, maptype):
        """
        Set the figure properties based on the map type.

        Args:
        - maptype: The type of map to plot ("sp" for Singapore, "nl" for the Netherlands).

        Returns:
        - figsize: The size of the figure.
        - figlims: The limits of the figure.
        - scale_factor: The scale factor for the figure.
        - map_name: The name of the map.
        """

        if maptype == "sp":
            figsize = (20, 10)
            figlims = (1232, 665)
            scale_factor = 35
            map_name = "Singapore"
        elif maptype == "nl":
            figsize = (7.5, 12)
            figlims = (475, 550)
            scale_factor = 600
            map_name = "the Netherlands"
        else:
            raise ValueError(f"Invalid map type: {maptype}")

        return figsize, figlims, scale_factor, map_name

    def plot_map(
        self,
        route=None,
        plot_nodes=False,
        show_axes=True,
        total_distance=None,
        save_name=False,
    ):
        """
        Plot the map with the edges and nodes.

        Args:
        - route: A list of node IDs representing the route to plot.
        - plot_nodes: A boolean indicating whether to plot the nodes.
        - show_axes: A boolean indicating whether to show the axes.
        - total_distance: The total distance of the route.
        - save_name: The name of the file to save the plot to.
        """

        plt.figure(figsize=self.figsize)

        for edge in self.edges:
            if (
                route
                and edge.source in route
                and (
                    (
                        route.index(edge.source) + 1 < len(route)
                        and edge.target == route[route.index(edge.source) + 1]
                    )
                    or (
                        route.index(edge.source) - 1 >= 0
                        and edge.target == route[route.index(edge.source) - 1]
                    )
                )
            ):

                color = "red"
                linewidth = 3
            else:
                color = "black"
                linewidth = 1
            plt.plot(
                [self.nodes[edge.source].coords[0], self.nodes[edge.target].coords[0]],
                [
                    self.figlims[1] - self.nodes[edge.source].coords[1],
                    self.figlims[1] - self.nodes[edge.target].coords[1],
                ],
                color=color,
                linewidth=linewidth,
            )

        if route:
            plt.scatter(
                self.nodes[route[0]].coords[0],
                self.figlims[1] - self.nodes[route[0]].coords[1],
                color="black",
                zorder=3,
                s=50,
            )
            plt.text(
                self.nodes[route[0]].coords[0] + 5,
                self.figlims[1] - self.nodes[route[0]].coords[1] + 5,
                "Start",
                fontsize=12,
            )
            plt.scatter(
                self.nodes[route[-1]].coords[0],
                self.figlims[1] - self.nodes[route[-1]].coords[1],
                color="black",
                zorder=3,
                s=50,
            )
            plt.text(
                self.nodes[route[-1]].coords[0] + 5,
                self.figlims[1] - self.nodes[route[-1]].coords[1] + 5,
                "End",
                fontsize=12,
            )

        if plot_nodes:
            for id, node in self.nodes.items():
                plt.scatter(
                    node.coords[0],
                    self.figlims[1] - node.coords[1],
                    color="black",
                    s=12,
                )
                plt.text(
                    node.coords[0] + 0.4,
                    self.figlims[1] - node.coords[1],
                    f"{id}",
                    fontsize=8,
                )

        if total_distance:
            plt.figtext(
                0.5,
                0.95,
                f"Total distance of route: {total_distance:.2f}",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=12,
            )

        if not show_axes:
            plt.axis("off")

        plt.xlim(0, self.figlims[0])
        plt.ylim(0, self.figlims[1])

        if save_name:
            plt.savefig(save_name)
        plt.show()
