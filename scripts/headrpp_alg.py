import random, math
import numpy as np
from tqdm import tqdm


class HEADRPP:
    def __init__(self, repr, nr_generations, start_node, end_node, population_size):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size

    def generate_route(self):
        route = [self.start_node]
        while route[-1] != self.end_node:
            forward_nodes = [
                node
                for node in self.repr.nodes[route[-1]].adjacent_nodes
                if node not in route
            ]
            if forward_nodes == []:
                return self.generate_route()
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

    def create_offspring(self, parents, p_crossover, p_mutation):
        offspring = []
        choices = range(len(parents))
        for _ in range(0, self.population_size, 2):
            p1 = parents[random.choice(choices)].copy()
            p2 = parents[random.choice(choices)].copy()
            # if random.random() <= p_c:
            #     p1, p2 = crossover(p1, p2)
            # if random.random() <= mu:
            #     p1 = mutation(p1)
            # if random.random() <= mu:
            #     p2 = mutation(p2)
            offspring.append(p1)
            offspring.append(p2)

        return offspring

    def run_algorithm(self):
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

        return population, avg_fitness, best_fitness
