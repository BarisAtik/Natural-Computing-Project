# Comparative Analysis of Genetic Algorithms for Route Planning

Welcome to the project repository of Group 14 for the Natural Computing course at Radboud University. This repository contains scripts, data, and instructions on how to reproduce the results of our experiments.

**Natural Computing project by Group 14 - Authors: Tygo Francissen, Dick Blankvoort, Baris Atik**

## Overview

In this project, the goal is to compare genetic algorithms of our own design. The first algorithm was inspired by HEADRPP, a genetic algorithm for dynamic route planning, while the second was inspired by the former along with the multi-objective algorithm NSGA-II. We compare the performance of these two algorithms both to each other and to Dijkstra's algorithm on maps of the Netherlands and Singapore in various route planning scenarios.

## Repository

This repository is structured in different categories including data, scripts, and notebooks.

### Notebooks

The notebooks are the starting points for the experiments. These notebooks include:

- `explorer.ipynb`: The main notebook used for running the experiments, plotting and data exploration. It being a notebook allows us to explore our representations in a more dynamic manner.
- `plotter.ipynb`: A small side notebook used to process the results into more sophisticated plots. Irrelevant when trying to reproduce our results.

### Scripts

The `scripts` folder contains the various scripts used in our implementation that are being used by the notebooks. These scripts consist of:

- `comparator.py`: provides a class to compare the HEADRPP and NSGA-II algorithms on two different maps for the multi-objective scenario.
- `dijkstra_alg.py`: contains our implementation of Dijkstra's algorithm.
- `framework.py`: provides the framework for our implementation, handling the reading of nodes and edges, the assignment of metrics to the graph, and the plotting of roads and networks.
- `genetic_algorithm.py`: contains functions that are used by both HEADRPP and NSGA-II, thereby serving as a parent of those classes.
- `headrpp_alg.py`: contains our implementation of the HEADRPP algorithm and is a subclass of `GeneticAlgorithm` to be able to use its functions.
- `NSGA2.py`: contains our implementation of the NSGA-II algorithm and is a subclass of `GeneticAlgorithm` to be able to use its functions.
- `single_objective.py`: provides a class to compare the HEADRPP, NSGA-II, and Dijkstra's algorithms on two different maps for the single-objective scenario.

### Data

The `data` folder contains the various data files used in our implementation. It contains .csv files storing the nodes and edges for our two map representations.

## Output Files

When running either the single-objective or multi-objective experiments, JSON files are stored to save the results. Additionally, figures of the plots are stored for all runs of an algorithm, which will end up in the `images` or `results` folder. The stored JSON files contain the results of the comparison between the algorithms on the maps of Singapore and the Netherlands for different experiments.

For the multi-objective experiments, each of the keys contains a list of results, where each result is a list of the following values for the best solution in each run:

- the fitness value of the route
- the total distance of the route
- the traffic score of the route
- the total pollution of the route
- the hotspot score of the route
- the time taken to run the algorithm

For the single-objective experiments, the JSON files are similar but split between maps and Dijkstra's results are added.

## Reproduction Instructions

In the case that you want to reproduce our experimental results, we refer to the Explorer notebook. This notebook contains the used parameters, routes, and libraries used in this study. It already contains the outputs when running all notebook cells after each other. To reproduce the experiments, the weights or routes can be changed in the indicated cells. For the single- and multi-objective experiments, note that only 1 run is currently displayed. If you want to do the original amount of runs, you have to comment out the indicated lines with `nr_runs=1`. Note that the running time increases to 10-20 hours per experiment.
