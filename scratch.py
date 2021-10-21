import array as arr
import pandas as pd
from deap import base
from deap import creator
from deap import tools


indir = '/home/drew/projects/iPSC_EA_Fitting_Sep2021/cell_2/EA_fit_092421/EA_output/run_1_092921/'

filename = indir + 'pop_final_093021_020021.txt'
pop_params = pd.read_csv(filename,delimiter=' ')
filename = indir + 'pop_strategy_093021_020021.txt'
pop_strategy = pd.read_csv(filename,delimiter=' ')
filename = indir + 'pop_fitness_093021_020021.txt'
pop_fitness = pd.read_csv(filename,delimiter=' ')
filename = indir + 'hof_093021_020021.txt'
hof_params = pd.read_csv(filename, delimiter=' ')
filename = indir + 'hof_fitness_093021_020021.txt'
hof_fitness = pd.read_csv(filename, delimiter=' ')


pop_ = (pop_params, pop_strategy, pop_fitness)
hof_ = (hof_params, hof_fitness)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", arr.array, typecode="d",
               fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", arr.array, typecode="d")

def rstrtES(ind_clss, strategy_clss, fit_clss, data):
    """This function constructs an individual from a prior EA population."""
    # Pass parameter to individual class
    ind = ind_clss(data[0])
    ind.strategy = strategy_clss(data[1])
    ind.fitness = fit_clss(data[2])

    return ind

def rstrtHOF(ind_clss, fit_clss, data):
    """This function constructs an individual from a prior EA population."""
    # Pass parameter to individual class
    ind = ind_clss(data[0])
    ind.fitness = fit_clss(data[1])

    return ind

def initRstrtPop(container, rstInd, pop_data):
    pop = []
    N = pop_data[0].shape[0]
    for i in range(N):
        ind_data = (list(pop_data[0].iloc[0, :]), list(pop_data[1].iloc[0, :]),
                    tuple(pop_data[2].iloc[0, :]))
        pop.append(rstInd(data=ind_data))
    return container(pop)

toolbox = base.Toolbox()

toolbox.register("individual", rstrtES, creator.Individual, creator.Strategy,
                 creator.FitnessMin, data=None)
toolbox.register("population", initRstrtPop, list, toolbox.individual, pop_)

pop = toolbox.population()
#hof = 
