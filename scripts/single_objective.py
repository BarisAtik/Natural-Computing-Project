import scripts.framework as fr
import scripts.headrpp_alg as headrpp
import scripts.dijkstra_alg as dijkstra
import scripts.NSGA2 as nsga2
import matplotlib.pyplot as plt
import time
import json


def calc_pollution(route, repr):
    """Calculate the pollution of a route."""
    total_pollution = 0
    for i in range(len(route)):
        total_pollution += repr.nodes[route[i]].pollution
    return total_pollution


def calc_traffic(route, repr):
    """Calculate the traffic of a route."""
    total_traffic = 0
    for i in range(len(route)):
        total_traffic += repr.nodes[route[i]].traffic
    return total_traffic


def calc_hotspots(route, repr):
    """Calculate the hotspots of a route."""
    total_hotspots = 0
    for i in range(len(route)):
        total_hotspots += repr.nodes[route[i]].hotspots
    return total_hotspots


def store_results(
    file_path,
    version,
    weight_name,
    results_headrpp,
    results_nsga2,
    results_dijkstra,
    country_name,
):
    """Store the results of the experiment in a json file."""
    with open(
        f"{file_path}/{version}_weight_{weight_name}_country_{country_name}.json", "w"
    ) as f:
        json.dump(
            {
                "results_headrpp": results_headrpp,
                "results_nsga2": results_nsga2,
                "results_dijkstra": results_dijkstra,
            },
            f,
        )


def run_single_objective_comp(
    version,
    repr,
    start_node,
    end_node,
    pop_size,
    nr_gen_headrpp,
    nr_gen_nsga2,
    nr_runs,
    country_name,
    weights,
    file_path="results",
    p_crossover=0.8,
    p_mutation=0.3,
    group_size=2,
):
    """
    Run the single objective comparison experiment.

    Args:
    - version: The version of the experiment.
    - repr: The representation of the map.
    - start_node: The start node of the route.
    - end_node: The end node of the route.
    - pop_size: The population size.
    - nr_gen_headrpp: The number of generations for HEADRPP.
    - nr_gen_nsga2: The number of generations for NSGA-II.
    - nr_runs: The number of runs.
    - country_name: The name of the country.
    - weights: The weights for the objectives.
    - file_path: The file path to store the results.
    - p_crossover: The probability of crossover.
    - p_mutation: The probability of mutation.
    - group_size: The group size for HEADRPP.
    """

    dijk = dijkstra.DIJKSTRA(repr, start_node, end_node)

    weight_names = ["distance", "traffic", "pollution", "hotspots"]

    i = 0
    if weights == [0, 1, 0, 0]:
        i = 1
    elif weights == [0, 0, 1, 0]:
        i = 2
    elif weights == [0, 0, 0, 1]:
        i = 3

    nsga = nsga2.NSGA2(
        repr,
        nr_gen_nsga2,
        start_node,
        end_node,
        pop_size,
        weights,
        p_crossover,
        p_mutation,
        group_size,
    )
    head = headrpp.HEADRPP(
        repr,
        nr_gen_headrpp,
        start_node,
        end_node,
        pop_size,
        weights,
        p_crossover,
        p_mutation,
        group_size,
    )

    nsga_costs = []
    headrpp_costs = []
    nsga_times = []
    headrpp_times = []
    dijkstra_costs = []
    dijkstra_results = []
    dijkstra_times = []

    # Run the algorithms for a number of runs
    for _ in range(nr_runs):
        start_time = time.time()
        nsga_cost = nsga.run_algorithm(show_results=False, show_progressbar=False)[
            2 * i + 6
        ][-1]
        nsga_costs.append(nsga_cost)
        nsga_times.append(time.time() - start_time)

        start_time = time.time()
        headrpp_cost = head.run_algorithm(show_results=False, show_progressbar=False)[
            2 * i + 6
        ][-1]
        headrpp_costs.append(headrpp_cost)
        headrpp_times.append(time.time() - start_time)

        start_time = time.time()
        best_route_dijkstra, avg_best_dijkstra = dijk.run_algorithm(
            show_results=False, weights=weights
        )
        if i == 0:
            pass
        elif i == 1:
            avg_best_dijkstra = calc_traffic(best_route_dijkstra, repr)
        elif i == 2:
            avg_best_dijkstra = calc_pollution(best_route_dijkstra, repr)
        elif i == 3:
            avg_best_dijkstra = calc_hotspots(best_route_dijkstra, repr)
        dijkstra_costs.append(avg_best_dijkstra)
        dijkstra_results.append(best_route_dijkstra)
        dijkstra_results.append(avg_best_dijkstra)
        dijkstra_times.append(time.time() - start_time)

    results_headrpp = {"costs": headrpp_costs, "times": headrpp_times}
    results_nsga2 = {"costs": nsga_costs, "times": nsga_times}
    results_dijkstra = {"costs": dijkstra_results, "times": dijkstra_times}

    # Store results after all runs
    store_results(
        file_path,
        version,
        weight_names[i],
        results_headrpp,
        results_nsga2,
        results_dijkstra,
        country_name,
    )

    # Plotting cost results
    fig, ax = plt.subplots()
    data = [nsga_costs, headrpp_costs, dijkstra_costs]
    ax.boxplot(data, labels=["NSGA-II", "HEADRPP", "Dijkstra"], showmeans=False)
    ax.set_ylabel("Cost")
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Cost Distribution for {weight_names[i]} on a map of {country_name}")

    save_name = "images/"
    alg_name = weight_names[i]
    plt.savefig(
        save_name
        + f"{version}_{country_name}_{alg_name.lower()}_{ax.get_ylabel().split('(')[0][:-1].replace(' ', '_').lower()}.png"
    )
    plt.show()

    # Plotting time results
    fig, ax = plt.subplots()
    time_data = [nsga_times, headrpp_times, dijkstra_times]
    ax.boxplot(time_data, labels=["NSGA-II", "HEADRPP", "Dijkstra"], showmeans=False)
    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Algorithm")
    ax.set_title(
        f"Execution Time Distribution for {weight_names[i]} on a map of {country_name}"
    )

    save_name = "images/"
    alg_name = weight_names[i]
    plt.savefig(
        save_name
        + f"{version}_{country_name}_{alg_name.lower()}_{ax.get_ylabel().split('(')[0][:-1].replace(' ', '_').lower()}.png"
    )
    plt.show()
