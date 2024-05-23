import scripts.framework as fr
import scripts.headrpp_alg as headrpp
import scripts.dijkstra_alg as dijkstra
import scripts.nsga2 as nsga2
import matplotlib.pyplot as plt
import time
import json

def calc_polution(route, repr):
    total_polution = 0
    for i in range(len(route) - 1):
        total_polution += repr.nodes[route[i]].pollution
    return total_polution

def calc_traffic(route, repr):
    total_traffic = 0
    for i in range(len(route) - 1):
        total_traffic += repr.nodes[route[i]].traffic
    return total_traffic

def calc_hotspots(route, repr):
    total_hotspots = 0
    for i in range(len(route) - 1):
        total_hotspots += repr.nodes[route[i]].hotspots
    return total_hotspots

def store_results(
    file_path, version, weight_name, results_headrpp, results_nsga2, results_dijkstra, country_name
):
    with open(f"{file_path}/weight_{weight_name}_version_{version}_country_{country_name}.json", "w") as f:
        json.dump(
            {
                "results_headrpp": results_headrpp,
                "results_nsga2": results_nsga2,
                "results_dijkstra": results_dijkstra,
            },
            f,
        )

def run_experiment_5(
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
    group_size=2
):
    # Dijkstra initialization
    dijk = dijkstra.DIJKSTRA(repr, start_node, end_node)

    weight_names = ["distance", "traffic", "pollution", "hotspots"]

    i = 0
    if weights == [0, 1, 0, 0]:
        i = 1
    elif weights == [0, 0, 1, 0]:
        i = 2
    elif weights == [0, 0, 0, 1]:
        i = 3

    nsga = nsga2.NSGA2(repr, nr_gen_nsga2, start_node, end_node, pop_size, weights, p_crossover, p_mutation, group_size)
    head = headrpp.HEADRPP(repr, nr_gen_headrpp, start_node, end_node, pop_size, weights, p_crossover, p_mutation, group_size)

    nsga_costs = []
    headrpp_costs = []
    nsga_times = []
    headrpp_times = []
    dijkstra_costs = []
    dijkstra_times = []

    for j in range(nr_runs):
        start_time = time.time()
        nsga_cost = nsga.run_algorithm(show_results=False, show_progressbar=False)[2 * i + 6][-1]
        nsga_costs.append(nsga_cost)
        nsga_times.append(time.time() - start_time)

        start_time = time.time()
        headrpp_cost = head.run_algorithm(show_results=False, show_progressbar=False)[2 * i + 6][-1]
        headrpp_costs.append(headrpp_cost)
        headrpp_times.append(time.time() - start_time)

        start_time = time.time()
        best_route_dijkstra, avg_best_dijkstra = dijk.run_algorithm(show_results=False, weights=weights)
        if i == 0:
            pass
        elif i == 1:
            avg_best_dijkstra = calc_traffic(best_route_dijkstra, repr)
            best_route_dijkstra = calc_traffic(best_route_dijkstra, repr)
        elif i == 2:
            avg_best_dijkstra = calc_polution(best_route_dijkstra, repr)
            best_route_dijkstra = calc_polution(best_route_dijkstra, repr)
        elif i == 3:
            avg_best_dijkstra = calc_hotspots(best_route_dijkstra, repr)
            best_route_dijkstra = calc_hotspots(best_route_dijkstra, repr)
        dijkstra_costs.append(best_route_dijkstra)
        dijkstra_costs.append(avg_best_dijkstra)
        dijkstra_times.append(time.time() - start_time)

    results_headrpp = {
        "costs": headrpp_costs,
        "times": headrpp_times
    }
    results_nsga2 = {
        "costs": nsga_costs,
        "times": nsga_times
    }
    results_dijkstra = {
        "costs": dijkstra_costs,
        "times": dijkstra_times
    }

    # Store results after all runs
    store_results(file_path, version, weight_names[i], results_headrpp, results_nsga2, results_dijkstra, country_name)

    # Plotting cost results
    fig, ax = plt.subplots()

    # Box plot for costs
    data = [nsga_costs, headrpp_costs, dijkstra_costs]
    ax.boxplot(data, labels=["NSGA-II", "HEADRPP", "Dijkstra"], showmeans=False)

    ax.set_ylabel("Cost")
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Cost Distribution for {weight_names[i]} on a map of {country_name}")

    # Saving the cost plot
    save_name = "images/avg_best"
    alg_name = weight_names[i]
    plt.savefig(
        save_name
        + f"_{alg_name.lower()}_{ax.get_ylabel().split('(')[0][:-1].replace(' ', '_').lower()}.png"
    )
    plt.show()

    # Plotting time results
    fig, ax = plt.subplots()

    # Box plot for times
    time_data = [nsga_times, headrpp_times, dijkstra_times]
    ax.boxplot(time_data, labels=["NSGA-II", "HEADRPP", "Dijkstra"], showmeans=False)

    ax.set_ylabel("Time (seconds)")
    ax.set_xlabel("Algorithm")
    ax.set_title(f"Execution Time Distribution for {weight_names[i]} on a map of {country_name}")

    # Saving the time plot
    save_name = "images/time_taken"
    alg_name = weight_names[i]
    plt.savefig(
        save_name
        + f"_{alg_name.lower()}_{ax.get_ylabel().split('(')[0][:-1].replace(' ', '_').lower()}.png"
    )
    plt.show()
