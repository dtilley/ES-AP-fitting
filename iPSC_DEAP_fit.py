import array as arr
import random
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import lognorm
from scipy.stats import loguniform
from run_dclamp_simulation import run_ind_dclamp
from cell_recording import ExperimentalAPSet
from multiprocessing import Pool

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def generateES(ind_clss, strategy_clss, size):
    """This function constructs an individual and its strategy.
    The regular paramters (ind) are sampled from the log-uniform distribution
    over the interval [0.01,5.0]. There is an additional parameter (phi)
    that corresponds to the leak of the injected dynamic-clamp current.
    Phi is sampled from from the uniform distribution and index is 0.
 
    The strategy object is the mutation strength of each paramter.
    It is sampled from the log-normal distribution."""

    # Construct the parameters in the individual
    phi = list(np.random.uniform(low=0.0, high=1.0, size=1))
    params = phi + list(loguniform.rvs(a=1e-2, b=5.0, size=(size-1)))
    ind = ind_clss(params)
 
    # Construct the strategy object.
    # The lognormal shape parameter is 0.5 just out of choice.
    phi = list(np.random.uniform(low=0.0, high=1.0, size=1))
    params = phi + list(lognorm.rvs(s=0.5, size=(size-1)))
    ind.strategy = strategy_clss(params)

    return ind


def fitness(ind, ExperAPSet):
    model_APSet = run_ind_dclamp(ind, dc_ik1=ExperAPSet.dc_ik1)
    rmsd_total = (sum(ExperAPSet.score(model_APSet).values()),)
    return rmsd_total


def mutateES(ind, indpb=0.3):
    for i in range(len(ind)):
        if (indpb > random.random()):
            # Mutate
            ind[i] *= lognorm.rvs(s=ind.strategy[i], size=1)
            ind.strategy[i] *= lognorm.rvs(s=ind.strategy[i], size=1)
    # Check that Phi is
    if (ind[0] > 1.0):
        # Reset
        ind[0] = random.random()
        ind.strategy[0] = random.random()
    return ind,


def cxESBlend(ind1, ind2, alpha):
    for i, (x1, s1, x2, s2) in enumerate(zip(ind1, ind1.strategy,
                                             ind2, ind2.strategy)):
        # Blend the values
        gamma = 1.0 - random.random() * alpha
        ind1[i] = gamma * x1 + (1.0 - gamma) * x2
        ind2[i] = gamma * x2 + (1.0 - gamma) * x1
        # Blend the strategies
        gamma = 1.0 - random.random() * alpha
        ind1.strategy[i] = (1. - gamma) * s1 + gamma * s2
        ind2.strategy[i] = gamma * s1 + (1. - gamma) * s2

    return ind1, ind2


# Add bash arguments eventually. main() #
def main():
    """This function applies the DEAP algorithm (mu,lambda) to fit
    the Kernik-Clancy model to an experimental AP data set.
    The 14 membrane conductance parameters are optimized.
    The fitness is defined as the sum of RMSD from each AP."""

    # Clock the start time.
    now = datetime.now()
    dt = now.strftime("%m%d%y_%H%M%S")
    print('Run start time: '+dt)
    
    NUM_PARAMS = 14

    PARAM_NAMES = ['phi', 'G_K1', 'G_Kr', 'G_Ks', 'G_to', 'P_CaL',
                   'G_CaT', 'G_Na', 'G_F', 'K_NaCa', 'P_NaK',
                   'G_b_Na', 'G_b_Ca', 'G_PCa']
    
    # Load in experimental AP set
    # Cell 2 recorded 12/24/20 Ishihara dynamic-clamp 1.0 pA/pF
    path_to_aps = '/home/drew/projects/iPSC_EA_Fitting_Sep2021/cell_2/AP_set'
    cell_2 = ExperimentalAPSet(path=path_to_aps, file_prefix='cell_2_',
                               file_suffix='_SAP.txt', cell_id=2, dc_ik1=1.0)

    # Define classes for EA with DEAP libaries. #
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", arr.array, typecode="d",
                   fitness=creator.FitnessMin, strategy=None)
    creator.create("Strategy", arr.array, typecode="d")

    # Create a toolbox to store the EA objects and functions.
    toolbox = base.Toolbox()

    # The (mu,lambda)_EA the toolbox must contain: mate, mutate, select, evaluate.
    # These functions allow the toolbox to populate a population with individuals.
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy, NUM_PARAMS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # These functions allow the population to evolve.
    toolbox.register("mate", cxESBlend, alpha=0.3)
    toolbox.register("mutate", mutateES)

    # Selection
    toolbox.register("evaluate", fitness, ExperAPSet=cell_2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Register some statistical functions to the toolbox.
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # To speed things up with multi-threading
    p = Pool()
    toolbox.register("map", p.map)

    #  Algorithm specific settings
    MU = 100  # Population size at the end of each generation including gen(0)
    LAMBDA = 150  # Number of new individuals generated per generation
    N_GEN = 10
    N_HOF = int((0.1) * MU * N_GEN)
    #N_HOF = 2

    hof = tools.HallOfFame(N_HOF)
    hof_fitness = []
    pop_fitness = []
    pop_strategy = []
    pop = toolbox.population(n=MU)
    pop_first_df = pd.DataFrame(pop, columns=PARAM_NAMES)
    pop_first_df.to_csv('pop_first_'+dt+'.txt', sep=' ', index=False)

    print('(mu,lambda): ('+str(MU)+','+str(LAMBDA)+')')
    
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=N_GEN, stats=stats,
                                              halloffame=hof, verbose=False)

    now = datetime.now()
    dt = now.strftime("%m%d%y_%H%M%S")
    print('Run end time: '+dt)
    logbook_df = pd.DataFrame(logbook)
    logbook_df.to_csv('logbook_'+dt+'.txt', sep=' ', index=False)

    pop_df = pd.DataFrame(pop, columns=PARAM_NAMES)
    pop_df.to_csv('pop_final_'+dt+'.txt', sep=' ', index=False)

    hof_df = pd.DataFrame(hof, columns=PARAM_NAMES)
    hof_df.to_csv('hof_'+dt+'.txt', sep=' ', index=False)

    for i in hof:
        hof_fitness.append(i.fitness.values[0])
    hof_fitness_pd = pd.DataFrame(hof_fitness, columns=["fitness"])
    hof_fitness_pd.to_csv('hof_fitness_'+dt+'.txt', sep=' ', index=False)

    for i in pop:
        pop_fitness.append(i.fitness.values[0])
        pop_strategy.append(i.strategy)
    pop_fitness_df = pd.DataFrame(pop_fitness, columns=["fitness"])
    pop_fitness_df.to_csv('pop_fitness_'+dt+'.txt', sep=' ', index=False)
    pop_strategy_df = pd.DataFrame(pop_strategy, columns=PARAM_NAMES)
    pop_strategy_df.to_csv('pop_strategy_'+dt+'.txt', sep=' ', index=False)

if __name__ == '__main__':
    main()
