import random


class HEADRPP:
    def __init__(self, repr, nr_generations, start_node, end_node, population_size):
        self.repr = repr
        self.nr_generations = nr_generations
        self.start_node = start_node
        self.end_node = end_node
        self.population_size = population_size

    def generate_route(self):
        route = [self.start_node]
        while route[-1] != self.end_node:
            forward_nodes = [
                node
                for node in self.repr.nodes[route[-1]].adjacent_nodes
                if node not in route
            ]
            if forward_nodes == []:
                return self.generate_route()
            next_node = random.choice(forward_nodes)
            route.append(next_node)
        return route

    def init_population(self):
        initial_population = []
        for _ in range(self.population_size):
            route = self.generate_route()
            initial_population.append(route)
        return initial_population

    def run_algorithm(self):
        population = self.init_population()
        # for pop in population:
        #     print(pop)
        return population
