import random, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class HEADRPP:
    def __init__(self, repr, nr_generations, start_node, end_node, population_size):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size

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
        total_distance = 0
        for i in range(0, len(route) - 1):
            total_distance += math.dist(
                self.repr.nodes[route[i]].coordinates,
                self.repr.nodes[route[i + 1]].coordinates,
            )
        return total_distance

    def tournament_selection(self, population, group_size):
        assert len(population) % group_size == 0

        random.shuffle(population)
        grouped_population = []
        for i in range(0, len(population), group_size):
            grouped_population.append(population[i : i + group_size])

        parents = []
        for group in grouped_population:
            winner = min(group, key=self.calculate_fitness)
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
        show_results=True,
        save_name=None,
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(range(nr_generations + 1), avg_fitness, label="Average for HEADRPP")
        plt.plot(range(nr_generations + 1), best_fitness, label="Best for HEADRPP")
        plt.xlim([0, nr_generations])
        plt.xlabel("Time (generations)")
        plt.ylabel("Route distance")
        plt.title(
            f"Fitness of population over time for the HEADRPP algorithm applied on a map of {map_name}"
        )
        plt.legend()
        if save_name:
            plt.savefig(save_name)
        if show_results:
            plt.show()
        plt.close()

    def run_algorithm(self, show_results=True, save_name=None):
        population = self.init_population()
        avg_fitness = [np.mean([self.calculate_fitness(route) for route in population])]
        best_fitness = [np.min([self.calculate_fitness(route) for route in population])]

        for _ in tqdm(range(self.nr_generations)):
            fittest_parents = self.tournament_selection(population, 2)
            population = self.create_offspring(fittest_parents, 0.8, 0.3)
            avg_fitness.append(
                np.mean([self.calculate_fitness(route) for route in population])
            )
            best_fitness.append(
                np.min([self.calculate_fitness(route) for route in population])
            )

        self.plot_results(
            avg_fitness,
            best_fitness,
            self.nr_generations,
            self.repr.map_name,
            show_results,
            save_name,
        )
        return population, avg_fitness, best_fitness
