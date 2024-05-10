import random, math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class NSGA2:
    def __init__(self, repr, nr_generations, start_node, end_node, population_size, weights):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size
        self.weights = weights
        self.max_distance, self.max_traffic, self.max_pollution, self.max_tourism = self.calculate_max_metrics()
        self.max_depth = 1000

    def calculate_max_metrics(self):
        max_distance = self.repr.scale_factor * sum(math.dist(
                    self.repr.nodes[edge.source].coords,
                    self.repr.nodes[edge.target].coords,
                )
                for edge in self.repr.edges
        )
        max_traffic = sum([node.traffic for node in self.repr.nodes.values()])
        max_pollution = sum([node.pollution for node in self.repr.nodes.values()])
        max_tourism = sum([node.tourist for node in self.repr.nodes.values()])

        return max_distance, max_traffic, max_pollution, max_tourism

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
        total_tourism = sum(self.repr.nodes[node].tourist for node in route)

        normalized_distance = total_distance / self.max_distance
        normalized_traffic = total_traffic / self.max_traffic
        normalized_pollution = total_pollution / self.max_pollution
        normalized_tourism = total_tourism / self.max_tourism

        fitness = (
            self.weights[0] * normalized_distance
            + self.weights[1] * normalized_traffic
            + self.weights[2] * normalized_pollution
            + self.weights[3] * normalized_tourism
        )
        return fitness, total_distance, total_traffic, total_pollution, total_tourism

    def fast_non_dominated_sort(self, population):
        frontiers = []
        dominated_solutions = [[] for _ in range(len(population))]
        num_dominations = [0] * len(population)

        for p_index, p in enumerate(population):
            p_fitness = self.calculate_fitness(p)
            for q_index, q in enumerate(population):
                q_fitness = self.calculate_fitness(q)
                if all(p_fitness[i] < q_fitness[i] for i in range(1, 4)):
                    dominated_solutions[p_index].append(q_index)
                    num_dominations[q_index] += 1

        frontiers.append([p_index for p_index in range(len(population)) if num_dominations[p_index] == 0])

        i = 0
        while frontiers[-1]:
            next_front = []
            for p_index in frontiers[-1]:
                for q_index in dominated_solutions[p_index]:
                    num_dominations[q_index] -= 1
                    if num_dominations[q_index] == 0:
                        next_front.append(q_index)
            frontiers.append(next_front)
            i += 1

        list_population_members = [[population[i] for i in front] for front in frontiers]

        return list_population_members

    def crowding_distance_selection(self, frontiers, group_size):
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
                        total_distance = 0
                        for l in range(0, len(frontiers[i][j]) - 1):
                            total_distance += math.dist(
                                self.repr.nodes[frontiers[i][j][l]].coords,
                                self.repr.nodes[frontiers[i][j][l + 1]].coords,
                            )
                        total_traffic = sum([self.repr.nodes[node].traffic for node in frontiers[i][j]])
                        total_pollution = sum([self.repr.nodes[node].pollution for node in frontiers[i][j]])

                        sum_distance_i_j = total_distance / self.max_distance
                        sum_traffic_i_j = total_traffic / self.max_traffic
                        sum_pollution_i_j = total_pollution / self.max_pollution

                        total_distance = 0
                        for l in range(0, len(frontiers[i][k]) - 1):
                            total_distance += math.dist(
                                self.repr.nodes[frontiers[i][k][l]].coords,
                                self.repr.nodes[frontiers[i][k][l + 1]].coords,
                            )
                        total_traffic = sum([self.repr.nodes[node].traffic for node in frontiers[i][k]])
                        total_pollution = sum([self.repr.nodes[node].pollution for node in frontiers[i][k]])

                        sum_distance_i_k = total_distance / self.max_distance
                        sum_traffic_i_k = total_traffic / self.max_traffic
                        sum_pollution_i_k = total_pollution / self.max_pollution
                        
                        # Calculate the crowding distance for the individual frontier[i][j]

                        coordinate_i_j = (sum_distance_i_j , sum_traffic_i_j, sum_pollution_i_j) # (sum_feature1, sum_feature2, sum_feature3)
                        coordinate_i_k = (sum_distance_i_k, sum_traffic_i_k, sum_pollution_i_k) # (sum_feature1, sum_feature2, sum_feature3)
                        distance_i_j_k = math.dist(coordinate_i_j, coordinate_i_k)
                        if closest_neighbour_1 is None:
                            closest_neighbour_1 = (k, distance_i_j_k)
                        elif closest_neighbour_2 is None:
                            closest_neighbour_2 = (k, distance_i_j_k)
                        else:
                            if distance_i_j_k < closest_neighbour_1[1]:
                                closest_neighbour_1 = (k, distance_i_j_k)
                            elif distance_i_j_k < closest_neighbour_2[1]:
                                closest_neighbour_2 = (k, distance_i_j_k)
                    # Now we calculate the crowding distance for this individual
                    # Which is the sum of the distances of the two closest neighbours
                    crowding_distance_i_j = closest_neighbour_1[-1] + closest_neighbour_2[-1]
                    crowding_distances.append((frontiers[i][j], crowding_distance_i_j))

                    # Sort crowding_distances based on crowding distances
                    crowding_distances.sort(key=lambda x: x[1], reverse=True)

                # To fill the population we need k more parents
                k = group_size - len(parents)
                k_parents = [pair[0] for pair in crowding_distances[:k]]
                parents.extend(k_parents)
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
        show_results=True,
        save_name=None,
    ):
        plt.figure(figsize=(10, 7))
        plt.plot(range(nr_generations + 1), avg_fitness, label="Average for NSGA2")
        plt.plot(range(nr_generations + 1), best_fitness, label="Best for NSGA2")
        plt.xlim([0, nr_generations])
        plt.xlabel("Time (generations)")
        plt.ylabel(ylabel)
        plt.title(
            f"{ylabel} of population over time for the NSGA2 algorithm applied on a map of {map_name}"
        )
        plt.legend()
        if save_name:
            plt.savefig(save_name + f"_nsga2_{ylabel.split('(')[0].replace(' ', '_').lower()}.png")
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
        avg_tourism, best_tourism = [np.mean(fitness_values[:, 4])], [
            np.min(fitness_values[:, 4])
        ]

        n = len(population)
        for _ in tqdm(range(self.nr_generations)):
            offspring = self.create_offspring(population, 0.8, 0.3)
            population.extend(offspring)
            frontiers = self.fast_non_dominated_sort(population)
            population = self.crowding_distance_selection(frontiers, n)

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
            avg_tourism.append(np.mean(fitness_values[:, 4]))
            best_tourism.append(np.min(fitness_values[:, 4]))

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
            "Route distance (in meters)",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_traffic,
            best_traffic,
            self.nr_generations,
            self.repr.map_name,
            "Route traffic (in 'cars encountered')",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_pollution,
            best_pollution,
            self.nr_generations,
            self.repr.map_name,
            "Route pollution (in μg/m^3)",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_tourism,
            best_tourism,
            self.nr_generations,
            self.repr.map_name,
            "Route tourism (in 'tourists encountered')",
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
            avg_tourism,
            best_tourism,
        )