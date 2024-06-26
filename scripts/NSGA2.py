import math
import numpy as np
from tqdm.auto import tqdm
from scripts.genetic_algorithm import GeneticAlgorithm


class NSGA2(GeneticAlgorithm):
    """
    A class to represent the NSGA2 algorithm.
    """

    def fast_non_dominated_sort(self, population):
        """
        Perform fast non-dominated sorting on the population.

        Args:
        - population: A list of routes.

        Returns:
        - list_population_members: A list of lists of routes, where each list represents a frontier.
        """

        frontiers = []
        dominated_solutions = [[] for _ in range(len(population))]
        num_dominations = [0] * len(population)

        for p_index, p in enumerate(population):
            p_fitness = self.calculate_fitness(p)
            for q_index, q in enumerate(population):
                q_fitness = self.calculate_fitness(q)
                if all(
                    p_fitness[i + 1] < q_fitness[i + 1]
                    for i in range(3)
                    if self.weights[i] > 0
                ) and (p_fitness[3] > q_fitness[3] or self.weights[3] == 0):
                    dominated_solutions[p_index].append(q_index)
                    num_dominations[q_index] += 1

        frontiers.append(
            [
                p_index
                for p_index in range(len(population))
                if num_dominations[p_index] == 0
            ]
        )

        # Now we will find the rest of the frontiers
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

        list_population_members = [
            [population[i] for i in front] for front in frontiers
        ]

        return list_population_members

    def crowding_distance_selection(self, frontiers, group_size):
        """
        Perform crowding distance selection on the frontiers to select the parents for the next generation.

        Args:
        - frontiers: A list of lists of routes, where each list represents a frontier.
        - group_size: The number of parents to select.

        Returns:
        - parents: A list of routes selected as parents.
        """

        parents = []
        i = 0
        while len(parents) < group_size:
            if (len(frontiers[i]) + len(parents)) <= group_size:
                parents.extend(frontiers[i])
            else:
                # Calculate the maximum values for each of the fitness values
                crowding_distances = []
                for j in range(0, len(frontiers[i])):
                    closest_neighbour_1 = None
                    closest_neighbour_2 = None
                    for k in range(0, len(frontiers[i])):
                        total_distance = 0
                        for l in range(0, len(frontiers[i][j]) - 1):
                            total_distance += math.dist(
                                self.repr.nodes[frontiers[i][j][l]].coords,
                                self.repr.nodes[frontiers[i][j][l + 1]].coords,
                            )
                        total_traffic = sum(
                            [self.repr.nodes[node].traffic for node in frontiers[i][j]]
                        )
                        total_pollution = sum(
                            [
                                self.repr.nodes[node].pollution
                                for node in frontiers[i][j]
                            ]
                        )
                        total_hotspots = sum(
                            [self.repr.nodes[node].hotspots for node in frontiers[i][j]]
                        )

                        sum_distance_i_j = total_distance / self.max_distance
                        sum_traffic_i_j = total_traffic / self.max_traffic
                        sum_pollution_i_j = total_pollution / self.max_pollution
                        sum_hotspots_i_j = 1 - total_hotspots / self.max_hotspots

                        # Calculate the crowding distance for the individual frontier[i][j]
                        total_distance = 0
                        for l in range(0, len(frontiers[i][k]) - 1):
                            total_distance += math.dist(
                                self.repr.nodes[frontiers[i][k][l]].coords,
                                self.repr.nodes[frontiers[i][k][l + 1]].coords,
                            )
                        total_traffic = sum(
                            [self.repr.nodes[node].traffic for node in frontiers[i][k]]
                        )
                        total_pollution = sum(
                            [
                                self.repr.nodes[node].pollution
                                for node in frontiers[i][k]
                            ]
                        )
                        total_hotspots = sum(
                            [self.repr.nodes[node].hotspots for node in frontiers[i][k]]
                        )

                        sum_distance_i_k = total_distance / self.max_distance
                        sum_traffic_i_k = total_traffic / self.max_traffic
                        sum_pollution_i_k = total_pollution / self.max_pollution
                        sum_hotspots_i_k = 1 - total_hotspots / self.max_hotspots

                        # Calculate the crowding distance for the individual frontier[i][j]
                        coordinate_i_j = (
                            self.weights[0] * sum_distance_i_j,
                            self.weights[1] * sum_traffic_i_j,
                            self.weights[2] * sum_pollution_i_j,
                            self.weights[3] * sum_hotspots_i_j,
                        )
                        coordinate_i_k = (
                            self.weights[0] * sum_distance_i_k,
                            self.weights[1] * sum_traffic_i_k,
                            self.weights[2] * sum_pollution_i_k,
                            self.weights[3] * sum_hotspots_i_k,
                        )
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

                    # Calculate the crowding distance for the individual frontier[i][j]
                    crowding_distance_i_j = (
                        closest_neighbour_1[-1] + closest_neighbour_2[-1]
                    )
                    crowding_distances.append((frontiers[i][j], crowding_distance_i_j))

                    crowding_distances.sort(key=lambda x: x[1], reverse=True)

                k = group_size - len(parents)
                k_parents = [pair[0] for pair in crowding_distances[:k]]
                parents.extend(k_parents)
            i = i + 1
        return parents

    def run_algorithm(self, show_results=True, save_name=None, show_progressbar=True):
        """
        Run the NSGA2 algorithm to find the best route.

        Args:
        - show_results: A boolean indicating whether to plot the results.
        - save_name: The name of the file to save the plots to.
        - show_progressbar: A boolean indicating whether to show the progress bar.

        Returns:
        - population: A list of routes in the final population.
        - best_route: The best route found by the algorithm.
        - best_route_values: The fitness values of the best route.
        - avg_fitness: A list of the average fitness values over the generations.
        - best_fitness: A list of the best fitness values over the generations.
        - avg_distance: A list of the average distance values over the generations.
        - best_distance: A list of the best distance values over the generations.
        - avg_traffic: A list of the average traffic values over the generations.
        - best_traffic: A list of the best traffic values over the generations.
        - avg_pollution: A list of the average pollution values over the generations.
        - best_pollution: A list of the best pollution values over the generations.
        - avg_hotspots: A list of the average hotspots values over the generations.
        - best_hotspots: A list of the best hotspots values over the generations.
        """

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
        avg_hotspots, best_hotspots = [np.mean(fitness_values[:, 4])], [
            np.max(fitness_values[:, 4])
        ]

        n = len(population)

        generations = (
            tqdm(range(self.nr_generations), desc="Running NSGA2")
            if show_progressbar
            else range(self.nr_generations)
        )

        for _ in generations:
            offspring = self.create_offspring(population)
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
            avg_hotspots.append(np.mean(fitness_values[:, 4]))
            best_hotspots.append(np.max(fitness_values[:, 4]))

        best_route = population[np.argmin(fitness_values[:, 0])]
        best_route_values = self.calculate_fitness(best_route)

        self.plot_results(
            avg_fitness,
            best_fitness,
            self.nr_generations,
            self.repr.map_name,
            "Overall fitness (weighted sum of distance, traffic, pollution, and hotspots)",
            "NSGA2",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_distance,
            best_distance,
            self.nr_generations,
            self.repr.map_name,
            "Route distance (in meters)",
            "NSGA2",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_traffic,
            best_traffic,
            self.nr_generations,
            self.repr.map_name,
            "Route traffic (in 'cars encountered')",
            "NSGA2",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_pollution,
            best_pollution,
            self.nr_generations,
            self.repr.map_name,
            "Route pollution (in μg/m^3)",
            "NSGA2",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_hotspots,
            best_hotspots,
            self.nr_generations,
            self.repr.map_name,
            "Route hotspots (in 'hotspot level')",
            "NSGA2",
            show_results,
            save_name,
        )

        return (
            population,
            best_route,
            best_route_values,
            avg_fitness,
            best_fitness,
            avg_distance,
            best_distance,
            avg_traffic,
            best_traffic,
            avg_pollution,
            best_pollution,
            avg_hotspots,
            best_hotspots,
        )
