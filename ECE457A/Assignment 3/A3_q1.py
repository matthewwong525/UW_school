import pandas as pd
import numpy as np
import random
import itertools


class TabuSearch():
    def __init__(self, iters=50, tabu_tenure=20, dynamic_tabu_tenure=None, aspiration_criteria=None, use_whole_neighbour=True, use_freq_mem=False, quiet=True):
        # grabs distance and flow from csv file
        self.dist = pd.read_csv('Distance.csv', header=None).to_numpy()
        self.flow = pd.read_csv('Flow.csv', header=None).to_numpy()
        assert self.dist.shape == self.flow.shape

        # calculates values and generates all possible combinations of i and j
        self.values = self.dist * self.flow
        self.n = self.values.shape[0]
        self.iters = iters
        i, j = np.where(np.triu(np.ones_like(self.values)) != 0)
        self.ij_list = list(zip(i, j))
        self.tabu_tenure = random.randint(*dynamic_tabu_tenure) if dynamic_tabu_tenure else tabu_tenure
        self.dynamic_tabu_tenure = dynamic_tabu_tenure
        self.aspiration_criteria = aspiration_criteria
        self.use_whole_neighbour = use_whole_neighbour
        self.use_freq_mem = use_freq_mem
        self.best_solu = None
        self.best_solu_cost = None
        self.quiet = quiet

    def tabu_search(self):
        # define initial solution
        # values are factory #, and indices are sites
        solu = np.arange(0, 20)
        random.shuffle(solu)

        tabu_struct = np.zeros(self.values.shape)
        self.best_solu = solu
        self.best_solu_cost = self.calc_total_cost(solu)

        # find neighbourhood
        for z in range(self.iters):
            tabu_i, tabu_j = np.where(tabu_struct != 0)
            # decrement tabu structure recency memory
            tabu_struct[tabu_i, tabu_j] -= 1

            # get neighbours and select candidate
            neighbours = self.get_neighbourhood()
            candidate = self.select_candidate(solu, tabu_struct, neighbours)
            i, j = candidate
            solu[i], solu[j] = solu[j], solu[i]
            cost = self.calc_total_cost(solu)
            if cost < self.best_solu_cost:
                self.best_solu = list(solu)
                self.best_solu_cost = cost

            tabu_struct[i, j] = self.tabu_tenure
            if self.dynamic_tabu_tenure and z % 10 == 0:
                self.tabu_tenure = random.randint(*self.dynamic_tabu_tenure)

            # increment tabu structure frequency memory
            tabu_struct[j, i] += 1

            if self.best_solu_cost == 1285:
                print(z)
                break

            if z % 100 == 0 and not self.quiet:
                print(self.best_solu_cost)

    # move operator selects random
    def calc_total_cost(self, solu):
        factories_1, factories_2 = np.where(np.triu(self.flow) != 0)
        total_cost = 0
        for fac_1, fac_2 in zip(factories_1, factories_2):
            fac_1_site = np.where(solu == fac_1)[0][0]
            fac_2_site = np.where(solu == fac_2)[0][0]
            total_cost += self.flow[fac_1, fac_2] * self.dist[fac_1_site, fac_2_site]

        return total_cost

    def get_neighbourhood(self):
        """
        Finds all swap combinations (i.e. neighbours) of the current solution
        """
        swap_combinations = list(itertools.combinations(range(self.n), 2))
        if self.use_whole_neighbour:
            return swap_combinations
        else:
            return random.sample(swap_combinations, 50)

    def select_candidate(self, solu, tabu_struct, neighbours):
        # perform all the swaps and calculate total cost of every
        best_candidate = None
        best_candidate_cost = None

        tabu_i, tabu_j = np.where(np.triu(tabu_struct) != 0)
        tabu_list = list(zip(tabu_i, tabu_j))
        penalization = 0

        for i, j in neighbours:
            temp_solu = list(solu)
            temp_solu[i], temp_solu[j] = temp_solu[j], temp_solu[i]
            temp_cost = self.calc_total_cost(temp_solu)

            # removes items from the tabu list that fit in the aspiration criteria
            if self.aspiration_criteria == 'best_sol' and (i, j) in tabu_list and temp_cost < self.best_solu_cost:
                tabu_list.remove((i, j))
            if self.aspiration_criteria == 'best_neigh_sol' and (i, j) in tabu_list and (best_candidate_cost is None or temp_cost < best_candidate_cost):
                tabu_list.remove((i, j))

            if self.use_freq_mem:
                penalization = tabu_struct[j, i]

            if (best_candidate_cost is None or (temp_cost + penalization) < best_candidate_cost) and ((i, j) not in tabu_list):
                best_candidate = (i, j)
                best_candidate_cost = temp_cost + penalization

        return best_candidate

    def get_solution(self):
        return self.best_solu, self.best_solu_cost


# Simple Tabu Search:
"""
random.seed(1)
x = TabuSearch()
x.tabu_search()
print(x.get_solution())
"""

# Change Initial Starting Point 10 times
"""
for i in range(10):
    random.seed(i)
    x = TabuSearch()
    x.tabu_search()
    print(x.get_solution())
"""

# Change the tabu list size twice:
"""
random.seed(1)
x = TabuSearch(tabu_tenure=5)
x.tabu_search()
print(x.get_solution())
random.seed(1)
y = TabuSearch(tabu_tenure=50)
y.tabu_search()
print(y.get_solution())
"""

# Change the tabu list size to a dynamic one:
"""
random.seed(1)
x = TabuSearch(dynamic_tabu_tenure=[5, 50])
x.tabu_search()
print(x.get_solution())
"""

# Add 2 aspiration criteria
"""
random.seed(1)
x = TabuSearch(tabu_tenure=20, aspiration_criteria='best_sol')
x.tabu_search()
print(x.get_solution())
random.seed(1)
y = TabuSearch(tabu_tenure=20, aspiration_criteria='best_neigh_sol')
y.tabu_search()
print(y.get_solution())
"""

# Use less than the whole neighbourhood
"""
random.seed(1)
x = TabuSearch(tabu_tenure=20, aspiration_criteria='best_sol', use_whole_neighbour=False)
x.tabu_search()
print(x.get_solution())
"""

# Use frequency based tabu list:
"""
random.seed(1)
x = TabuSearch(iters=5000, tabu_tenure=20, aspiration_criteria='best_sol', use_freq_mem=True, use_whole_neighbour=True, quiet=False)
x.tabu_search()
print(x.get_solution())
"""
