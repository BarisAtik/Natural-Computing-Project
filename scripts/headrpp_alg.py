import random, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class HEADRPP:
    def __init__(
        self, repr, nr_generations, start_node, end_node, population_size, weights
    ):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size
        self.weights = weights
        self.max_distance = max(
            [
                math.dist(node1.coordinates, node2.coordinates)
                for node1 in self.repr.nodes.values()
                for node2 in self.repr.nodes.values()
            ]
        )
        self.max_traffic = max([node.traffic for node in self.repr.nodes.values()])
        self.max_pollution = max([node.pollution for node in self.repr.nodes.values()])

    def generate_route(self, start_node=None, old_route=[]):
        route = [start_node] if start_node else [self.start_node]
        while route[-1] != self.end_node:
            forward_nodes = [
                node
                for node in self.repr.nodes[route[-1]].adjacent_nodes
                if node not in route and node not in old_route
            ]
            if forward_nodes == []:
                return self.generate_route(start_node, old_route)
            next_node = random.choice(forward_nodes)
            route.append(next_node)
        return route

    def init_population(self):
        initial_population = []
        for _ in range(self.population_size):
            route = self.generate_route()
            initial_population.append(route)
        return initial_population

    def calculate_fitness(self, route):
        total_distance, min_distance, max_distance = 0, np.Inf, 0
        for i in range(0, len(route) - 1):
            distance = math.dist(
                self.repr.nodes[route[i]].coordinates,
                self.repr.nodes[route[i + 1]].coordinates,
            )
            total_distance += distance
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance

        traffic = [self.repr.nodes[node].traffic for node in route]
        total_traffic, min_traffic, max_traffic = (
            sum(traffic),
            min(traffic),
            max(traffic),
        )
        pollution = [self.repr.nodes[node].pollution for node in route]
        total_pollution, min_pollution, max_pollution = (
            sum(pollution),
            min(pollution),
            max(pollution),
        )

        # normalized_distance = (total_distance - min_distance) / max_distance
        normalized_distance = total_distance / self.max_distance
        # normalized_traffic = (total_traffic - min_traffic) / max_traffic
        normalized_traffic = total_traffic / self.max_traffic
        # normalized_pollution = (total_pollution - min_pollution) / max_pollution
        normalized_pollution = total_pollution / self.max_pollution

        fitness = (
            self.weights[0] * normalized_distance
            + self.weights[1] * normalized_traffic
            + self.weights[2] * normalized_pollution
        )
        return fitness, total_distance, total_traffic, total_pollution

    def tournament_selection(self, population, group_size):
        assert len(population) % group_size == 0

        random.shuffle(population)
        grouped_population = []
        for i in range(0, len(population), group_size):
            grouped_population.append(population[i : i + group_size])

        parents = []
        for group in grouped_population:
            winner = min(group, key=lambda x: self.calculate_fitness(x)[0])
            parents.append(winner)
        return parents

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
        mutated_route = route[:index] + self.generate_route(
            mutation_point, route[:index]
        )
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
        show_results=True,
        save_name=None,
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(range(nr_generations + 1), avg_fitness, label="Average for HEADRPP")
        plt.plot(range(nr_generations + 1), best_fitness, label="Best for HEADRPP")
        plt.xlim([0, nr_generations])
        plt.xlabel("Time (generations)")
        plt.ylabel(ylabel)
        plt.title(
            f"{ylabel} of population over time for the HEADRPP algorithm applied on a map of {map_name}"
        )
        plt.legend()
        if save_name:
            plt.savefig(save_name + f"_headrpp_{ylabel.replace(' ', '_').lower()}.png")
        if show_results:
            plt.show()
        plt.close()

    def run_algorithm(self, show_results=True, save_name=None):
        population = self.init_population()

        fitness_values = np.array(
            [self.calculate_fitness(route) for route in population]
        )

        avg_fitness, best_fitness = [np.mean(fitness_values[:, 0])], [
            np.min(fitness_values[:, 0])
        ]
        avg_distance, best_distance = [np.mean(fitness_values[:, 1])], [
            np.min(fitness_values[:, 1])
        ]
        avg_traffic, best_traffic = [np.mean(fitness_values[:, 2])], [
            np.min(fitness_values[:, 2])
        ]
        avg_pollution, best_pollution = [np.mean(fitness_values[:, 3])], [
            np.min(fitness_values[:, 3])
        ]

        for _ in tqdm(range(self.nr_generations)):
            fittest_parents = self.tournament_selection(population, 2)
            population = self.create_offspring(fittest_parents, 0.8, 0.3)
            fitness_values = np.array(
                [self.calculate_fitness(route) for route in population]
            )
            avg_fitness.append(np.mean(fitness_values[:, 0]))
            best_fitness.append(np.min(fitness_values[:, 0]))
            avg_distance.append(np.mean(fitness_values[:, 1]))
            best_distance.append(np.min(fitness_values[:, 1]))
            avg_traffic.append(np.mean(fitness_values[:, 2]))
            best_traffic.append(np.min(fitness_values[:, 2]))
            avg_pollution.append(np.mean(fitness_values[:, 3]))
            best_pollution.append(np.min(fitness_values[:, 3]))

        best_route = population[np.argmin(fitness_values[:, 0])]

        self.plot_results(
            avg_fitness,
            best_fitness,
            self.nr_generations,
            self.repr.map_name,
            "Fitness",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_distance,
            best_distance,
            self.nr_generations,
            self.repr.map_name,
            "Route distance",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_traffic,
            best_traffic,
            self.nr_generations,
            self.repr.map_name,
            "Route traffic",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_pollution,
            best_pollution,
            self.nr_generations,
            self.repr.map_name,
            "Route pollution",
            show_results,
            save_name,
        )

        return (
            population,
            best_route,
            avg_fitness,
            best_fitness,
            avg_distance,
            best_distance,
            avg_traffic,
            best_traffic,
            avg_pollution,
            best_pollution,
        )
