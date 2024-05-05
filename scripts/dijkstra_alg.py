import numpy as np
import matplotlib.pyplot as plt


class DIJKSTRA:
    def __init__(self, repr, start_node, end_node):
        self.repr = repr
        self.start_node = start_node
        self.end_node = end_node

    def plot_results(self, path, total_distance, save_name=False):
        # Plot the results
        self.repr.plot_map(path, plot_nodes=True, total_distance=total_distance, save_name=save_name)

    def run_algorithm(self, show_results=True, save_name=False):
        # Initialize the distance and predecessor dictionaries
        distances = {node: float("inf") for node in self.repr.nodes}
        preds = {node: None for node in self.repr.nodes}
        distances[self.start_node] = 0

        # Initialize the set of visited nodes
        visited = set()

        # Run the algorithm
        while visited != set(self.repr.nodes):
            # Find the node with the smallest distance
            unvisited = {node: distances[node] for node in distances if node not in visited}
            current_node = min(unvisited, key=unvisited.get)
            visited.add(current_node)

            # Update the distances and predecessors of the adjacent nodes
            for node in self.repr.nodes[current_node].adjacent_nodes:
                dist = np.linalg.norm(np.array(self.repr.nodes[node].coordinates) - np.array(self.repr.nodes[current_node].coordinates))
                if distances[node] > distances[current_node] + dist:
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
            total_distance += np.linalg.norm(np.array(self.repr.nodes[path[i]].coordinates) - np.array(self.repr.nodes[path[i+1]].coordinates))
        
        # Plot the results
        if show_results:
            self.plot_results(path, total_distance, save_name)

        return path, total_distance

