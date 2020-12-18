import matlab.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eng = matlab.engine.connect_matlab()
np.random.seed(1)

def call_perfFCN(K_p, T_i, T_d):
    vector=matlab.double([K_p, T_i, T_d])
    column_vector=eng.transpose(vector)
    return eng.perfFCN(column_vector, nargout=4)

def calc_fitness(row):
    matlab_ans = call_perfFCN(row['k_p'], row['t_i'], row['t_d'])
    fitness = sum(1/np.array(matlab_ans))
    return np.nan_to_num(fitness)

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

def crossover(df, crossover_prob = 0.6):
    samples = df.shape[0]
    # shuffles parents
    parent_df = df.sample(frac=1).reset_index(drop=True)
    
    roll = np.random.rand(samples//2)
    p_1_indices = np.argwhere(roll <= crossover_prob).flatten() * 2 + 1
    p_2_indices = p_1_indices - 1
    all_crossover_indices = list(p_1_indices) + list(p_2_indices)

    # whole arithmetic crossover
    alpha = 0.5

    # get parents
    parents_1 = df[['k_p', 't_i', 't_d']].iloc[p_1_indices].reset_index(drop=True)
    parents_2 = df[['k_p', 't_i', 't_d']].iloc[p_2_indices].reset_index(drop=True)

    # get children
    children_1 = (alpha * parents_1) + ((1-alpha) * parents_2)
    children_2 = (alpha * parents_2) + ((1-alpha) * parents_1)

    # concatenates all children
    children_df = pd.concat([children_1, children_2]).reset_index(drop=True)
    children_df['fitness'] = pd.NA

    return parent_df, children_df, all_crossover_indices

def mutation(df, k_p_range=(2, 18), t_i_range=(1.05, 9.42), t_d_range=(0.26, 2.37), mutation_prob = 0.25):
    samples = df.shape[0]
    roll = np.random.rand(samples, 3)
    flip_loc = np.argwhere(roll <= mutation_prob).transpose()

    k_p_indices = flip_loc[0][np.where(flip_loc[1] == 0)]
    t_i_indices = flip_loc[0][np.where(flip_loc[1] == 1)]
    t_d_indices = flip_loc[0][np.where(flip_loc[1] == 2)]

    df.loc[k_p_indices, 'k_p'] = np.random.rand(len(k_p_indices))*(k_p_range[1] - k_p_range[0]) + k_p_range[0]
    df.loc[t_i_indices, 't_i'] = np.random.rand(len(t_i_indices))*(t_i_range[1] - t_i_range[0]) + t_i_range[0]
    df.loc[t_d_indices, 't_d'] = np.random.rand(len(t_d_indices))*(t_d_range[1] - t_d_range[0]) + t_d_range[0]

    df['fitness'] = pd.NA

    return df


def survivor_selection(parent_df, children_df, non_mating_df, crossover_indices):
    samples = parent_df.shape[0]
    parent_df['generation'] = 'parent'
    children_df['generation'] = 'child'
    non_mating_df['generation'] = 'single'

    # concatenate all parents and children, then calculate fitness of all
    all_popu_df = pd.concat([parent_df, children_df, non_mating_df]).reset_index()
    all_popu_df.loc[all_popu_df['fitness'].isnull(), 'fitness'] = all_popu_df.loc[all_popu_df['fitness'].isnull()].apply(calc_fitness, axis=1)
    
    # select top 2 candidates (elitism)
    elite_df = all_popu_df.sort_values('fitness', ascending=False).head(2)
    # parents that were not part of the crossover and not part of the elite (NOT NEEDED)
    parent_df = all_popu_df.loc[(~all_popu_df['index'].isin(crossover_indices)) & (all_popu_df['generation'] == 'parent') & (~all_popu_df.index.isin(elite_df.index))]
    
    # parents that did not mate and are not part of the elite
    non_mating = all_popu_df.loc[(all_popu_df['generation'] == 'single') & (~all_popu_df.index.isin(elite_df.index))]
    # children that are not part of the elite
    new_children_df = all_popu_df.loc[(all_popu_df['generation'] == 'child') & (~all_popu_df.index.isin(elite_df.index))]
    # gathers top values to fill the rest of the population
    remaining_df = pd.concat([non_mating, new_children_df]).sort_values('fitness', ascending=False).head(samples - 2)

    # collect new population
    df = pd.concat([elite_df, remaining_df]).drop(['generation', 'index'], axis=1).reset_index(drop=True)
    return df

def plot(best_fitness_list):
    plt.plot(best_fitness_list)
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()

def genetic_alg(samples=50, generations=150, k_p_range=(2, 18), t_i_range=(1.05, 9.42), t_d_range=(0.26, 2.37)):
    representation = []
    rand_arr = np.random.rand(samples,3)
    
    # initialize 50 population
    popu_df = pd.DataFrame(rand_arr, columns=['k_p', 't_i', 't_d'])
    popu_df['k_p'] = popu_df['k_p']*(k_p_range[1] - k_p_range[0]) + k_p_range[0]
    popu_df['t_i'] = popu_df['t_i']*(t_i_range[1] - t_i_range[0]) + t_i_range[0]
    popu_df['t_d'] = popu_df['t_d']*(t_d_range[1] - t_d_range[0]) + t_d_range[0]

    # calculate fitness
    popu_df['fitness'] = popu_df.apply(calc_fitness, axis=1)
    best_fitness_list = [popu_df['fitness'].max()]

    for i in range(generations):
        mating_df, non_mating_df = parent_selection(popu_df)
        parent_df, children_df, crossover_indices = crossover(mating_df, 0.6)
        children_df = mutation(children_df, k_p_range, t_i_range, t_d_range, 0.5)
        popu_df = survivor_selection(parent_df, children_df, non_mating_df, crossover_indices)
        best_fitness_list.append(popu_df['fitness'].max())
        #print(popu_df.iloc[popu_df['fitness'].argmax()])

    plot(best_fitness_list)
    


genetic_alg(samples=50, generations=150)
