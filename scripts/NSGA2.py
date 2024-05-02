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
        total_polution = 0
        # Decide a way to calculate the total polution
        total_time = 0
        # For each distance, calculate the time it takes to travel that distance according to speed limit
        return 1 * total_distance + 1 * total_polution + 1 * total_time

    def fast_non_dominated_sort(self, population, group_size):
        frontiers = []
        dominated_solutions = {sol: [] for sol in population}
        num_dominations = {sol: 0 for sol in population}
        # dominated_solutions is a dictionary
        # key values, now we can save for each instance in the population, by whom it is dominated.
        assert len(population) % group_size == 0
        for p in population:
            for q in population:
                p_fitness = self.calculate_fitness(p)
                q_fitness = self.calculate_fitness(q)
                if p_fitness < q_fitness:
                    dominated_solutions[p].append(q)
                elif p_fitness > q_fitness:
                    num_dominations[p] += 1
        frontiers[1] = [p for p in population if num_dominations[p] == 0]

        i = 1
        while frontiers[i] != []:
            next_front = []
            for p in frontiers[i]:
                for q in dominated_solutions[p]:
                    num_dominations[q] -= 1
                    if num_dominations[q] == 0:
                        next_front.append(q)
            i += 1
            frontiers[i] = next_front
        return frontiers

    def crowding_distance_selection(self, frontiers, group_size):
        assert len(frontiers) % group_size == 0

        parents = []
        i = 0
        while(len(parents)<group_size):
            if ((len(frontiers[i])+len(parents)) <= group_size):
                parents.extend(frontiers[i])
            else:
                # Now we select the one's with the biggest crowding distance
                # We calculate the crowding distance for each individual in this frontier
                # We sort the individuals in this frontier based on the crowding distance
                # We select the first individuals until we have the group size
                crowding_distances = []
                for j in range(0, len(frontiers[i])):
                    # Now calculate the crowding distance for the individual frontier[i][j]
                    # We will do this by finding the two closest neighbours to frontier[i][j] in this front
                    closest_neighbour_1 = None
                    closest_neighbour_2 = None
                    for k in range(0, len(frontiers[i])):
                        # Calculate the crowding distance for the individual frontier[i][j]
                        if closest_neighbour_1 is None:
                            distance_1 = math.dist(frontiers[i][j], frontiers[i][k])
                            closest_neighbour_1 = (k, distance_1)
                        elif closest_neighbour_2 is None:
                            distance_2 = math.dist(frontiers[i][j], frontiers[i][k])
                            closest_neighbour_2 = (k, distance_2)
                        else:
                            distance_3 = math.dist(frontiers[i][j], frontiers[i][k])
                            if distance_3 < closest_neighbour_1[1]:
                                closest_neighbour_1 = (k, distance_3)
                            elif distance_3 < closest_neighbour_2[1]:
                                closest_neighbour_2 = (k, distance_3)
                    # Now we calculate the crowding distance for this individual
                    # Which is the sum of the distances of the two closest neighbours
                    crowding_distance_i_j = closest_neighbour_1[-1] + closest_neighbour_2[-1]
                    crowding_distances.append((frontiers[i][j], crowding_distance_i_j))
                # To fill the population we need k more parents
                k = group_size - len(parents)
                k_parents = crowding_distances[:k]
                for j in range(0, k):
                    parents.append(k_parents[j][0])
            i = i + 1
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
        # child1 = parent1[:]
        # child2 = parent2[:]
        # for i in range(len(parent1)):
        #     if random.random() < 0.5:
        #         child1[i], child2[i] = child2[i], child1[i]
        # return child1, child2
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