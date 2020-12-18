import numpy as np
import copy
import pandas as pd
from anytree import Node, RenderTree, PostOrderIter
import matplotlib.pyplot as plt


def calc_value(bin_string):
    ind = int(bin_string[:2], 2)
    return bin_string[2:][ind] == '1'

def get_multiplex_mapping():
    mapping = [calc_value('{0:06b}'.format(i)) for i in range(64)]
    return np.array(mapping)

def parent_selection(df):
    samples = df.shape[0]

    # use fitness proportionate selection to select parent
    fitness_sum = df['fitness'].sum()
    fitness_prob = df['fitness'] / fitness_sum

    # roulette wheel algorithm
    roulette_list = [np.random.choice(samples, p=fitness_prob) for i in range(samples)]
    mating_df = df.iloc[roulette_list]
    non_mating_df = df.loc[~df.index.isin(roulette_list)].copy()
    return mating_df, non_mating_df

def get_random_node(head):
    return copy.deepcopy(np.random.choice(head.descendants))

def mutation(df, mutation_prob = 0.25):
    samples = df.shape[0]
    roll = np.random.rand(samples)
    flip_loc = np.argwhere(roll <= mutation_prob).flatten()

    for i in flip_loc:
        rand_tree = get_random_func_tree()
        rand_node = get_random_node(df.loc[i, 'tree'])
        rand_tree.parent = rand_node.parent
        rand_node.parent = None
        df.loc[i, 'tree'] = rand_tree.root 

    df['fitness'] = pd.NA
    return df

def crossover(df, crossover_prob = 0.6):
    samples = df.shape[0]
    # shuffles parents
    parent_df = df.sample(frac=1).reset_index(drop=True)
    
    roll = np.random.rand(samples//2)
    p_1_indices = np.argwhere(roll <= crossover_prob).flatten() * 2 + 1
    p_2_indices = p_1_indices - 1
    all_crossover_indices = list(p_1_indices) + list(p_2_indices)

    children_df = parent_df.copy(deep=True)
    for i1, i2 in zip(p_1_indices, p_2_indices):
        rand_node1 = get_random_node(children_df.loc[i1, 'tree'])
        rand_node2 = get_random_node(children_df.loc[i2, 'tree'])
        rand_node1.parent, rand_node2.parent = rand_node2.parent, rand_node1.parent
        children_df.loc[i1, 'tree'] = rand_node1.root
        children_df.loc[i2, 'tree'] = rand_node2.root

    # concatenates all children
    children_df = children_df.reset_index(drop=True)
    children_df['fitness'] = pd.NA

    return parent_df, children_df

def survivor_selection(samples, true_mapping, parent_df, children_df, non_mating_df):
    parent_df['generation'] = 'parent'
    children_df['generation'] = 'child'
    non_mating_df['generation'] = 'single'

    # concatenate all parents and children, then calculate fitness of all
    all_popu_df = pd.concat([parent_df, children_df, non_mating_df]).reset_index()
    all_popu_df.loc[all_popu_df['fitness'].isnull(), 'fitness'] = all_popu_df.loc[all_popu_df['fitness'].isnull()].apply(lambda row: calc_fitness(row['tree'], true_mapping), axis=1)
    
    # select top 2 candidates (elitism)
    elite_df = all_popu_df.sort_values('fitness', ascending=False).head(2)
    # parents that did not mate and are not part of the elite
    non_mating = all_popu_df.loc[(all_popu_df['generation'] == 'single') & (~all_popu_df.index.isin(elite_df.index))]
    # children that are not part of the elite
    new_children_df = all_popu_df.loc[(all_popu_df['generation'] == 'child') & (~all_popu_df.index.isin(elite_df.index))]
    # gathers top values to fill the rest of the population
    remaining_df = pd.concat([non_mating, new_children_df]).sort_values('fitness', ascending=False).head(samples - 2)

    # collect new population
    df = pd.concat([elite_df, remaining_df]).drop(['generation', 'index'], axis=1).reset_index(drop=True)
    return df


def evaluate_func_tree(head, bin_string):
    """
    converts the tree function into a string
    """
    func_set = ['&', '|', '!', '~']
    term_set = ['a0', 'a1', 'd0', 'd1', 'd2', 'd3']
    bool_arr = [c == '1' for c in bin_string]
    term_dict = dict(zip(term_set, bool_arr))
    term_dict[False] = False
    term_dict[True] = True
    head = copy.deepcopy(head)

    stack = []
    for node in PostOrderIter(head):
        if stack and node.name in func_set:
            temp = []
            operation = node.name
            while stack and stack[-1].parent == node:
                temp.insert(0, stack.pop().name)

            if operation == '&':
                node.name = np.all(temp)
            elif operation ==  '|':
                node.name = np.any(temp)
            elif operation == '!':
                assert len(temp) == 1
                node.name = not temp[0]
            elif operation == '~': #represents if
                assert len(temp) == 3
                node.name = temp[1] if temp[0] else temp[2]
            stack.append(node)
        else:
            node.name = term_dict[node.name]
            stack.append(node)

    return stack[0].name

def calc_fitness(head, true_mapping):
    """
    Finds all matches with XNOR logic
    """
    #print(evaluate_func_tree(head, '{0:06b}'.format(63)))
    mapping = [ evaluate_func_tree(head, '{0:06b}'.format(i)) for i in range(len(true_mapping)) ]
    xnor_logic = ~(mapping ^ true_mapping)
    return np.sum(xnor_logic)

    
def get_random_func_tree(func_p=0.2, term_p=0.8):
    func_set = ['&', '|', '!', '~']
    term_set = ['a0', 'a1', 'd0', 'd1', 'd2', 'd3']
    all_set = func_set + term_set
    #p = [func_p / len(func_set)] * len(func_set) + [term_p / len(term_set)] * len(term_set) np.random.choice(all_set)
    head = Node(np.random.choice(func_set))
    queue = [head]
    while queue:
        token = queue.pop(0)
        if token.name in func_set:
            if token.name in '&|':
                queue.extend([Node(x, parent=token) for x in np.random.choice(all_set, 2)])
            elif token.name == '!':
                queue.append(Node(np.random.choice(all_set), parent=token))
            elif token.name == '~':
                queue.extend([Node(x, parent=token) for x in np.random.choice(all_set, 3)])
    return head

def plot(best_fitness_list):
    plt.plot(best_fitness_list)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

def gp(generations=100, popu=50):
    true_mapping = get_multiplex_mapping()
    funcs = [get_random_func_tree() for i in range(popu)]
    fitness = [calc_fitness(f, true_mapping) for f in funcs]
    popu_df = pd.DataFrame({'tree': funcs, 'fitness': fitness})

    best_fitness_list = [popu_df['fitness'].max()]
    best_tree_list = [popu_df.loc[pd.to_numeric(popu_df['fitness']).argmax(), 'tree']]

    for i in range(generations):
        #print(RenderTree(popu_df.loc[pd.to_numeric(popu_df['fitness']).argmax(), 'tree']))
        print(popu_df['fitness'].max())
        mating_df, non_mating_df = parent_selection(popu_df)
        parents_df, children_df = crossover(mating_df, 0.6)
        children_df = mutation(children_df, 0.25)
        popu_df = survivor_selection(popu, true_mapping, parents_df, children_df, non_mating_df)
        best_fitness_list.append(popu_df['fitness'].max())
        best_tree_list.append(popu_df.loc[pd.to_numeric(popu_df['fitness']).argmax(), 'tree'])
    print(RenderTree(best_tree_list[-1]))
    plot(best_fitness_list)

gp(generations=50)
