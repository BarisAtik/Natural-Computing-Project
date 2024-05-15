import random
import numpy as np
from tqdm import tqdm
from scripts.genetic_algorithm import GeneticAlgorithm


class HEADRPP(GeneticAlgorithm):
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
        avg_hotspots, best_hotspots = [np.mean(fitness_values[:, 4])], [
            np.max(fitness_values[:, 4])
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
            avg_hotspots.append(np.mean(fitness_values[:, 4]))
            best_hotspots.append(np.max(fitness_values[:, 4]))

        best_route = population[np.argmin(fitness_values[:, 0])]

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
