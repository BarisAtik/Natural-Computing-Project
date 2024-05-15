import random, math
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(
        self, repr, nr_generations, start_node, end_node, population_size, weights
    ):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size
        self.weights = weights
        self.max_distance, self.max_traffic, self.max_pollution, self.max_hotspots = (
            self.calculate_max_metrics()
        )
        self.max_depth = 1000

    def calculate_max_metrics(self):
        max_distance = self.repr.scale_factor * sum(
            math.dist(
                self.repr.nodes[edge.source].coords,
                self.repr.nodes[edge.target].coords,
            )
            for edge in self.repr.edges
        )
        max_traffic = sum([node.traffic for node in self.repr.nodes.values()])
        max_pollution = sum([node.pollution for node in self.repr.nodes.values()])
        max_hotspots = sum([node.hotspots for node in self.repr.nodes.values()])

        return max_distance, max_traffic, max_pollution, max_hotspots

    def generate_route(self, start_node=None, old_route=[], depth=1):
        if depth > self.max_depth:
            return []

        route = [start_node] if start_node else [self.start_node]
        while route[-1] != self.end_node:
            forward_nodes = [
                node
                for node in self.repr.nodes[route[-1]].adjacent_nodes
                if node not in route and node not in old_route
            ]
            if forward_nodes == []:
                return self.generate_route(start_node, old_route, depth + 1)
            next_node = random.choice(forward_nodes)
            route.append(next_node)
        return route

    def init_population(self):
        initial_population = []
        while len(initial_population) < self.population_size:
            route = self.generate_route()
            if route != []:
                initial_population.append(route)
        return initial_population

    def calculate_fitness(self, route):
        total_distance = self.repr.scale_factor * sum(
            math.dist(
                self.repr.nodes[route[i]].coords,
                self.repr.nodes[route[i + 1]].coords,
            )
            for i in range(len(route) - 1)
        )
        total_traffic = sum(self.repr.nodes[node].traffic for node in route)
        total_pollution = sum(self.repr.nodes[node].pollution for node in route)
        total_hotspots = sum(self.repr.nodes[node].hotspots for node in route)

        normalized_distance = total_distance / self.max_distance
        normalized_traffic = total_traffic / self.max_traffic
        normalized_pollution = total_pollution / self.max_pollution
        normalized_hotspots = total_hotspots / self.max_hotspots

        fitness = (
            self.weights[0] * normalized_distance
            + self.weights[1] * normalized_traffic
            + self.weights[2] * normalized_pollution
            + self.weights[3] * (1 - normalized_hotspots)
        )
        return fitness, total_distance, total_traffic, total_pollution, total_hotspots

    def loop_anneal(self, route):
        visited = set()
        new_route = []
        for node in route:
            if node not in visited:
                visited.add(node)
                new_route.append(node)
            else:
                index = new_route.index(node)
                new_route = new_route[: index + 1]
                visited = set(new_route)
        return new_route

    def crossover(self, parent1, parent2):
        common_nodes = set(parent1) & set(parent2)
        if len(common_nodes) > 0:
            crossover_node = random.choice(list(common_nodes))
            index1 = parent1.index(crossover_node)
            index2 = parent2.index(crossover_node)
            child1 = parent1[:index1] + parent2[index2:]
            child2 = parent2[:index2] + parent1[index1:]
            return child1, child2
        else:
            return parent1, parent2

    def mutation(self, route):
        mutation_point = random.choice(route[1:-1])
        index = route.index(mutation_point)
        new_part = self.generate_route(mutation_point, route[:index])
        if new_part == []:
            return route
        mutated_route = route[:index] + new_part
        return mutated_route

    def create_offspring(self, parents, p_crossover, p_mutation):
        offspring = []
        choices = range(len(parents))
        for _ in range(0, self.population_size, 2):
            p1 = parents[random.choice(choices)].copy()
            p2 = parents[random.choice(choices)].copy()
            if random.random() <= p_crossover:
                p1, p2 = self.crossover(p1, p2)
                p1 = self.loop_anneal(p1)
                p2 = self.loop_anneal(p2)
            if random.random() <= p_mutation:
                p1 = self.mutation(p1)
                p1 = self.loop_anneal(p1)
            if random.random() <= p_mutation:
                p2 = self.mutation(p2)
                p2 = self.loop_anneal(p2)
            offspring.append(p1)
            offspring.append(p2)

        return offspring

    def plot_results(
        self,
        avg_fitness,
        best_fitness,
        nr_generations,
        map_name,
        ylabel,
        alg_name,
        show_results=True,
        save_name=None,
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(
            range(nr_generations + 1), avg_fitness, label=f"Average for {alg_name}"
        )
        plt.plot(range(nr_generations + 1), best_fitness, label=f"Best for {alg_name}")
        plt.xlim([0, nr_generations])
        plt.xlabel("Time (generations)")
        plt.ylabel(ylabel)
        plt.title(
            f"{ylabel.split('(')[0][:-1]} of population over time for the {alg_name} algorithm applied on a map of {map_name}"
        )
        plt.legend()
        if save_name:
            plt.savefig(
                save_name
                + f"_{alg_name.lower()}_{ylabel.split('(')[0][:-1].replace(' ', '_').lower()}.png"
            )
        if show_results:
            plt.show()
        plt.close()
