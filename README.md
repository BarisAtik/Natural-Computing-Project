# Natural computing project

This repository contains the files for the Natural Computing project of team 14.

- ./explorer.ipynb is the main notebook used for plotting and data exploration. It being a notebook allows us to explore our representation in a more dynamic manner.
- ./scripts/ contains the various scripts used in our implementation.
  - ./scripts/framework.py provides the framework for our implementation, handling the reading of nodes and edges, the assignment of metrics to the graph, and the plotting of roads and networks.
- ./data/ contains the various data files used in our implementation. Currently it contains .csv files storing the nodes and edges for our two map representations.

## JSON Files

The JSON files contain the results of the comparison between HEADRPP and NSGA-II on the maps of Singapore and the Netherlands for different experiments. Each of the keys contains a list of results, where each result is a list of the following values for the best solution in each run:

- the fitness value of the route
- the total distance of the route
- the traffic score of the route
- the total pollution of the route
- the hotspot score of the route
- the time taken to run the algorithm
