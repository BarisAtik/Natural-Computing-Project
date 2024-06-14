import random
import numpy as np
from tqdm.auto import tqdm
from scripts.genetic_algorithm import GeneticAlgorithm


class HEADRPP(GeneticAlgorithm):
    """
    A class to represent HEADRPP.
    """

    def tournament_selection(self, population):
        """
        Perform tournament selection to select the parents for the next generation.

        Args:
        - population: A list of routes.

        Returns:
        - parents: A list of routes selected as parents.
        """

        assert len(population) % self.group_size == 0

        random.shuffle(population)
        grouped_population = []
        for i in range(0, len(population), self.group_size):
            grouped_population.append(population[i : i + self.group_size])

        parents = []
        for group in grouped_population:
            winner = min(group, key=lambda x: self.calculate_fitness(x)[0])
            parents.append(winner)
        return parents

    def run_algorithm(self, show_results=True, save_name=None, show_progressbar=True):
        """
        Run the HEADRPP algorithm to find the best route.

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

        generations = (
            tqdm(range(self.nr_generations), desc="Running HEADRPP")
            if show_progressbar
            else range(self.nr_generations)
        )

        for _ in generations:
            fittest_parents = self.tournament_selection(population)
            population = self.create_offspring(fittest_parents)
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
            "HEADRPP",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_distance,
            best_distance,
            self.nr_generations,
            self.repr.map_name,
            "Route distance (in meters)",
            "HEADRPP",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_traffic,
            best_traffic,
            self.nr_generations,
            self.repr.map_name,
            "Route traffic (in 'cars encountered')",
            "HEADRPP",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_pollution,
            best_pollution,
            self.nr_generations,
            self.repr.map_name,
            "Route pollution (in μg/m^3)",
            "HEADRPP",
            show_results,
            save_name,
        )
        self.plot_results(
            avg_hotspots,
            best_hotspots,
            self.nr_generations,
            self.repr.map_name,
            "Route hotspots (in 'hotspots score')",
            "HEADRPP",
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
