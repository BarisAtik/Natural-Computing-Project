import numpy as np


class DIJKSTRA:
    """
    A class to represent the Dijkstra algorithm for finding the shortest path between two nodes.

    Attributes:
    - repr: The Representation object representing the network.
    - start_node: The ID of the start node.
    - end_node: The ID of the end node.
    """

    def __init__(self, repr, start_node, end_node):
        self.repr = repr
        self.start_node = start_node
        self.end_node = end_node

    def run_algorithm(self, weights=[1, 0, 0, 0], show_results=True, save_name=False):
        """
        Run the Dijkstra algorithm to find the shortest path between two nodes.

        Args:
        - weights: A list of weights for the distance, traffic, pollution, and hotspots metrics.
        - show_results: A boolean indicating whether to plot the results.
        - save_name: The name of the file to save the plot to.

        Returns:
        - path: A list of node IDs representing the shortest path.
        - total_distance: The total distance of the path.
        """

        # Initialize the distance and predecessor dictionaries
        distances = {node: float("inf") for node in self.repr.nodes}
        preds = {node: None for node in self.repr.nodes}
        distances[self.start_node] = 0

        # Initialize the set of visited nodes
        visited = set()

        # Run the algorithm
        while visited != set(self.repr.nodes):
            # Find the node with the smallest distance
            unvisited = {
                node: distances[node] for node in distances if node not in visited
            }
            current_node = min(unvisited, key=unvisited.get)
            visited.add(current_node)

            # Update the distances and predecessors of the adjacent nodes
            for node in self.repr.nodes[current_node].adjacent_nodes:
                dist = (
                    weights[0]
                    * self.repr.scale_factor
                    * np.linalg.norm(
                        np.array(self.repr.nodes[node].coords)
                        - np.array(self.repr.nodes[current_node].coords)
                    )
                    + weights[1] * self.repr.nodes[node].traffic
                    + weights[2] * self.repr.nodes[node].pollution
                    + weights[3] * (1 - self.repr.nodes[node].hotspots)
                )

                if (
                    distances[node] > distances[current_node] + dist
                    and preds[current_node] != node
                ):
                    distances[node] = distances[current_node] + dist
                    preds[node] = current_node

        # Reconstruct the path
        path = []
        current_node = self.end_node
        while current_node is not None:
            path.insert(0, current_node)
            current_node = preds[current_node]

        # Calculate the total distance
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += (
                weights[0]
                * self.repr.scale_factor
                * np.linalg.norm(
                    np.array(self.repr.nodes[path[i]].coords)
                    - np.array(self.repr.nodes[path[i + 1]].coords)
                )
                + weights[1] * self.repr.nodes[path[i]].traffic
                + weights[2] * self.repr.nodes[path[i]].pollution
                + weights[3] * self.repr.nodes[path[i]].hotspots
            )

        # Plot the results
        if show_results:
            self.repr.plot_map(
                path,
                plot_nodes=True,
                total_distance=total_distance,
                save_name=save_name,
            )

        return path, total_distance
