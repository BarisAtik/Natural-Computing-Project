import scripts.framework as fr
from tqdm.auto import tqdm
from scripts.headrpp_alg import HEADRPP
from scripts.nsga2 import NSGA2
from scripts.dijkstra_alg import DIJKSTRA
import matplotlib.pyplot as plt
from os import listdir
import time, json


class Comparator_with_dijkstra:
    def __init__(
        self,
        pop_size_sp,
        pop_size_nl,
        nr_gen_headrpp,
        nr_gen_nsga2,
        nr_gen_dijkstra,
        weights,
        p_crossover=0.8,
        p_mutation=0.3,
        group_size=2,
    ):
        self.pop_size_sp = pop_size_sp
        self.pop_size_nl = pop_size_nl
        self.nr_gen_headrpp = nr_gen_headrpp
        self.nr_gen_nsga2 = nr_gen_nsga2
        self.nr_gen_dijkstra = nr_gen_dijkstra
        self.weights = weights
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.group_size = group_size
        self.init_representations()
        self.init_run_number()

    def init_representations(self):
        self.repr_sp = fr.Representation(
            "./data/nodes_sp.csv", "./data/edges_sp.csv", maptype="sp"
        )
        self.repr_nl = fr.Representation(
            "./data/nodes_nl.csv", "./data/edges_nl.csv", maptype="nl"
        )

    def init_run_number(self):
        self.run_number = 1
        for file in listdir("results"):
            if "comp" in file:
                run_number = int(file.split("_")[1])
                if run_number >= self.run_number:
                    self.run_number = run_number + 1

    def evaluate_algorithm(
        self,
        algorithm,
        representation,
        nr_generations,
        start_node,
        end_node,
        population_size,
    ):
        if algorithm == DIJKSTRA:
            alg = algorithm(representation, start_node, end_node)
            start_time = time.time()
            result = alg.run_algorithm(weights=self.weights, show_results=False)[0]
            time_taken = time.time() - start_time
            return result + (time_taken,)
        else:
            alg = algorithm(
                representation,
                nr_generations,
                start_node,
                end_node,
                population_size,
                self.weights,
                self.p_crossover,
                self.p_mutation,
                self.group_size,
            )
            start_time = time.time()
            result = alg.run_algorithm(show_results=False, show_progressbar=False)[2]
            time_taken = time.time() - start_time
            return result + (time_taken,)

    def plot_results(self, map_name, results_headrpp, results_nsga2, results_dijkstra):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Comparison of HEADRPP, NSGA-II and Dijkstra on a map of {map_name}")
        objectives = [
            (i + 1, obj)
            for i, obj in enumerate(["Distance", "Traffic", "Pollution", "Hotspots"])
        ]
        x_axis = 0
        y_axis = 0

        for i, (index1, obj1) in enumerate(objectives):
            for j, (index2, obj2) in enumerate(objectives):
                if i >= j:
                    continue
                ax = axs[x_axis, y_axis]
                y_axis += 1
                if y_axis == 3:
                    x_axis += 1
                    y_axis = 0
                ax.set_title(f"{obj1} vs {obj2}")
                ax.set_xlabel(obj1)
                ax.set_ylabel(obj2)

                # Extract the values of the objectives from the results
                obj1_values_headrpp = [result[index1] for result in results_headrpp]
                obj2_values_headrpp = [result[index2] for result in results_headrpp]
                obj1_values_nsga2 = [result[index1] for result in results_nsga2]
                obj2_values_nsga2 = [result[index2] for result in results_nsga2]
                obj1_values_dijkstra = [result[index1] for result in results_dijkstra]
                obj2_values_dijkstra = [result[index2] for result in results_dijkstra]

                ax.scatter(
                    obj1_values_headrpp,
                    obj2_values_headrpp,
                    label="HEADRPP",
                    color="blue",
                )
                ax.scatter(
                    obj1_values_nsga2,
                    obj2_values_nsga2,
                    label="NSGA-II",
                    color="red",
                )
                ax.scatter(
                    obj1_values_dijkstra,
                    obj2_values_dijkstra,
                    label="Dijkstra",
                    color="green",
                )
                ax.legend()

        plt.tight_layout()
        plt.savefig(
            f"results/exp_{self.run_number}_comp_obj_{map_name.replace(' ', '_').lower()}.png"
        )
        plt.show()
        plt.close()

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Fitness against objectives on a map of {map_name}")
        for i, (index, obj) in enumerate(objectives):
            ax = axs[i // 2, i % 2]
            ax.set_title(f"Fitness vs {obj}")
            ax.set_xlabel("Fitness")
            ax.set_ylabel(obj)

            obj_values_headrpp = [result[index] for result in results_headrpp]
            obj_values_nsga2 = [result[index] for result in results_nsga2]
            obj_values_dijkstra = [result[index] for result in results_dijkstra]
            fitness_values_headrpp = [result[0] for result in results_headrpp]
            fitness_values_nsga2 = [result[0] for result in results_nsga2]
            fitness_values_dijkstra = [result[0] for result in results_dijkstra]


            ax.scatter(
                fitness_values_headrpp,
                obj_values_headrpp,
                label="HEADRPP",
                color="blue",
            )
            ax.scatter(
                fitness_values_nsga2, obj_values_nsga2, label="NSGA-II", color="red"
            )
            ax.scatter(
                fitness_values_dijkstra,
                obj_values_dijkstra,
                label="Dijkstra",
                color="green",
            )
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            f"results/exp_{self.run_number}_comp_fit_{map_name.replace(' ', '_').lower()}.png"
        )
        plt.show()
        plt.close()

        fig, ax = plt.subplots()
        ax.boxplot(
            [result[-1] for result in results_headrpp],
            positions=[1],
            labels=["HEADRPP"],
            patch_artist=True,
        )
        ax.boxplot(
            [result[-1] for result in results_nsga2],
            positions=[2],
            labels=["NSGA-II"],
            patch_artist=True,
        )
        ax.boxplot(
            [result[-1] for result in results_dijkstra],
            positions=[3],
            labels=["Dijkstra"],
            patch_artist=True,
        )
        ax.set_title(f"Time taken to run the algorithms on a map of {map_name}")
        ax.set_ylabel("Time taken (s)")
        plt.savefig(
            f"results/exp_{self.run_number}_comp_time_{map_name.replace(' ', '_').lower()}.png"
        )
        plt.show()
        plt.close()

    def store_results(
        self, results_headrpp_sp, results_headrpp_nl, results_nsga2_sp, results_nsga2_nl, results_dijkstra_sp, results_dijkstra_nl
    ):
        with open(f"results/exp_{self.run_number}_comp_data.json", "w") as f:
            json.dump(
                {
                    "results_headrpp_sp": results_headrpp_sp,
                    "results_headrpp_nl": results_headrpp_nl,
                    "results_nsga2_sp": results_nsga2_sp,
                    "results_nsga2_nl": results_nsga2_nl,
                    "results_dijkstra_sp": results_dijkstra_sp,
                    "results_dijkstra_nl": results_dijkstra_nl,
                },
                f,
            )

    def run_comparison(
        self, nr_runs, start_node_sp, end_node_sp, start_node_nl, end_node_nl
    ):
        results_headrpp_sp = []
        results_headrpp_nl = []
        results_nsga2_sp = []
        results_nsga2_nl = []
        results_dijkstra_sp = []
        results_dijkstra_nl = []
        for _ in tqdm(range(nr_runs), desc="Running comparison"):
            results_headrpp_sp.append(
                self.evaluate_algorithm(
                    HEADRPP,
                    self.repr_sp,
                    self.nr_gen_headrpp,
                    start_node_sp,
                    end_node_sp,
                    self.pop_size_sp,
                )
            )
            results_headrpp_nl.append(
                self.evaluate_algorithm(
                    HEADRPP,
                    self.repr_nl,
                    self.nr_gen_headrpp,
                    start_node_nl,
                    end_node_nl,
                    self.pop_size_nl,
                )
            )
            results_nsga2_sp.append(
                self.evaluate_algorithm(
                    NSGA2,
                    self.repr_sp,
                    self.nr_gen_nsga2,
                    start_node_sp,
                    end_node_sp,
                    self.pop_size_sp,
                )
            )
            results_nsga2_nl.append(
                self.evaluate_algorithm(
                    NSGA2,
                    self.repr_nl,
                    self.nr_gen_nsga2,
                    start_node_nl,
                    end_node_nl,
                    self.pop_size_nl,
                )
            )
            results_dijkstra_sp.append(
                self.evaluate_algorithm(
                    DIJKSTRA,
                    self.repr_sp,
                    self.nr_gen_dijkstra,
                    start_node_sp,
                    end_node_sp,
                    self.pop_size_sp,
                )
            )
            results_dijkstra_nl.append(
                self.evaluate_algorithm(
                    DIJKSTRA,
                    self.repr_nl,
                    self.nr_gen_dijkstra,
                    start_node_nl,
                    end_node_nl,
                    self.pop_size_nl,
                )
            )
        self.plot_results(self.repr_sp.map_name, results_headrpp_sp, results_nsga2_sp, results_dijkstra_sp)
        self.plot_results(self.repr_nl.map_name, results_headrpp_nl, results_nsga2_nl, results_dijkstra_nl)
        self.store_results(
            results_headrpp_sp, results_headrpp_nl, results_nsga2_sp, results_nsga2_nl, results_dijkstra_sp, results_dijkstra_nl
        )