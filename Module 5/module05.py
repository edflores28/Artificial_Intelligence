import sys
from random import choice
from random import randrange
from random import random
from random import sample
from copy import deepcopy
from random import gauss, random
from collections import deque
from decimal import Decimal
from io import StringIO

def convert_bin(individual):
    '''
    This routine converts the binary individual
    to its corresponding float valued list

    Args:
        indiviual - The list of binary values
    Returns:
        the individual in floating values
    '''
    values = []
    indiv_deque = deque(deepcopy(individual))
    while len(indiv_deque) > 0:
        binary = [indiv_deque.popleft() for x in range(10)]
        value = int(''.join(str(x) for x in binary), 2)
        values.append((float(value) - 512.00) / 100)
    return values

def generate_population(size, function, minimization, is_binary):
    '''
    This routine generates a random population

    Args:
        size - the total population size
        function - the shift sphere function
        minimization - flag indicating minimization
    Returns:
        the population with fitness scores
    '''
    TOTAL_VARIABLES = 10
    # Generator to generate float values in specific range
    def float_range(start, stop, increment):
        start = Decimal(start)
        increment = Decimal(increment)
        while start < stop:
            yield float(start)
            start += increment
    population = []
    values = []
    multiplier = 1
    if is_binary:
        values = [0, 1]
        multiplier = TOTAL_VARIABLES
    else:
        values = list(float_range(-5.12, 5.12, Decimal('0.01')))
    while len(population) < size:
        individual = deepcopy([choice(values) for x in range(TOTAL_VARIABLES*multiplier)])
        score = []
        if is_binary:
            score = fitness(convert_bin(individual), function, minimization)
        else:
            score = fitness(individual, function, minimization)
        population.append([score] + [individual])
    return population

def select_parents(population, total):
    '''
    This routine selects parents from the population

    Args:
        population - The population
        total - the total for tournament selection
    Returns
        parent_1 and parent_2
    '''
    # Pick total number of parents
    parents = sample(range(len(population)), total)
    # Set best and second best variables to 0
    best_value = -9999
    best_index = 0
    second_value = -9999
    second_index = 0
    # Iterate over the selected parents and
    # find the best one
    for x in parents:
        if population[x][0] > best_value:
            best_value = population[x][0]
            best_index = x
    # Iterate over the selected parents and
    # find the second best one
    for x in parents: 
        if population[x][0] > second_value and population[x][0] < best_value:
                second_value = population[x][0]
                second_index = x
    # Put the indices in a list and sort it.
    indices = [best_index, second_index]
    indices.sort()
    parent_2 = population.pop(indices[1])
    parent_1 = population.pop(indices[0])
    return parent_1, parent_2
           
def crossover(parent_1, parent_2, rate):
    '''
    This routine creates offspring from parents
    based on the reate

    Args:
        parent_1 - the first parent
        parent_2 - the second parent
        rate - the crossover rate
    Returns:
        child_1 and child_2 based on the rate
        parent_1 and parent_2 otherwise
    '''
    # Generate a random number and if less than
    # the rate, get a cross over point and
    # create offsping
    if random() <= rate:
        # Generate a crossover point
        index = randrange(len(parent_1[1])) + 1
        # Create children based on the crossover point
        child_1 = deepcopy(parent_2[1][0:index] + parent_1[1][index:])
        child_2 = deepcopy(parent_1[1][0:index] + parent_2[1][index:])
        return child_1, child_2
    # When greater than the rate return the parents
    else:
        return parent_1[1], parent_2[1]

def mutate_single(individual, rate, is_binary, sigma):
    '''
    This routine mutates a single individual

    Args:
        individual - the individual
        rate - the mutation rate
        is_binary - flag to indicare binary ga
        signa - std. dev. value for gauss distribution
    Returns:
        A mutated individual
    '''
    length = len(individual)
    for x in range(length):
        if random() <= rate/length:
            if is_binary:
                individual[x] = 1 if individual[x] == 0 else 0
            else:
                individual[x] += gauss(0.0, sigma)
                if individual[x] > 5.12:
                    individual[x] = 5.12
                if individual[x] < -5.12:
                    individual[x] = -5.12
    return individual

def mutate(child_1, child_2, rate, is_binary, sigma=0.0):
    '''
    This routine mutates the children

    Args:
        child_1 - the first child
        child_2 - the second child
        rate - the rate of mutation
    Returns:
        The offspring
    '''
    child_1 = mutate_single(child_1, rate, is_binary, sigma)
    child_2 = mutate_single(child_2, rate, is_binary, sigma)
    return child_1, child_2

def fitness(individual, function, minimization):
    '''
    This routine calculates the fitness score

    Args:
        individual - the individual to calculate
        function - the shift sphere function
    Returns:
        The fitness score
    '''
    score = function(individual)
    if minimization:
        return 1 / (1 - score)
    else:
        return score

def calculate_scores(child_1, child_2, function, minimization, is_binary):
    '''
    This routine calculates the fitness score of the offspring

    Args:
        child_1 - the first child
        child_2 - the second child
    Returns:
        returns the offspring with the fitness score
    '''
    if is_binary:
        fitness_1 = fitness(convert_bin(child_1), function, minimization)
        fitness_2 = fitness(convert_bin(child_2), function, minimization)
    else:
        fitness_1 = fitness(child_1, function, minimization)
        fitness_2 = fitness(child_2, function, minimization)
    return [fitness_1] + [child_1], [fitness_2] + [child_2]

def get_best_fit(population):
    '''
    This routine finds the best fit individual
    
    Args:
        population - the entire population
    Returns:
        Nothing
    '''
    best = -9999
    best_index = 0
    for indv in range(len(population)):
        if population[indv][0] > best:
            best = population[indv][0] 
            best_index = indv
    return deepcopy(population[best_index])

def print_params(parameters, is_binary):
    '''
    This routine prints the parameters

    Args:
        parametes - the parameters dictionary
        is_binary - flag indicating binary ga
    Returns:
        Nothing
    '''
    print("Total Population Size:", parameters['population'])
    print("Number of Generations:", parameters['generations'])
    print("Tournament Selection Total:", parameters['tournament_total'])
    print("Crossover Rate:", parameters['crossover_rate'])
    print("Mutation Rate:", parameters['mutation_rate'])
    if not is_binary:
        print("Guassian Sigma:", parameters['sigma'])

def print_indv(individual, function, is_binary, debug):
    '''
    This routine prints the best individual
    '''
    values = (individual[1])
    if is_binary:
        print("Encoding:", ''.join(str(x) for x in values))
        values = convert_bin(values)
    print("Values:", ' '.join(str(x) for x in values))
    if debug:
        print("Fitness Score:", individual[0])
        print("Shifted Sphere Value:", function(values))
    print()

def generic_algorithm(parameters, debug, is_binary):
    '''
    This genetic algorithm

    Args: 
        parameters - parameters dictionary
        debug - debug flag
        is_binary - flag to perform binary ga
    Returns:
        Nothing
    '''
    # Create a random population
    population = generate_population(parameters['population'], parameters['f'], parameters['minimization'], is_binary)
    # Iterate for a total generations
    for generation in range(parameters['generations']):
        # Select two parents
        parent_1, parent_2 = select_parents(population, parameters['tournament_total'])
        # Create child from the parents
        child_1, child_2 = crossover(parent_1, parent_2, parameters['crossover_rate'])
        # Mutate the children
        child_1, child_2 = mutate(child_1, child_2, parameters['mutation_rate'], is_binary, parameters['sigma'])
        # Calculate the fitness scores for the children
        child_1, child_2, = calculate_scores(child_1, child_2, parameters['f'], parameters['minimization'], is_binary)
        # Add the children back to the population
        population.append(child_1)
        population.append(child_2)
        # Only print the best individual on the last generation
        # if debug is False
        if not debug and generation == parameters['generations'] - 1:
            print("\nGenetic Algorithm Complete")
            print("Best Fit Individual:")
            print_indv(get_best_fit(population), parameters['f'], is_binary, debug)
            
        if debug:
            print("Generation", generation + 1)
            print_indv(get_best_fit(population), parameters['f'], is_binary, debug)

def binary_ga(parameters, debug):
    print_params(parameters, True)
    generic_algorithm(parameters, debug, True)

def real_ga(parameters, debug):
    print_params(parameters, False)
    generic_algorithm(parameters, debug, False)

def shifted_sphere( shift, xs):
    return sum( [(x - shift)**2 for x in xs])

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True,
        "crossover_rate": 0.9,
        "mutation_rate": 0.05,
        "generations": 1000,
        "tournament_total": 20,
        "population": 800,
        "sigma": 0
    }
    print("Executing Binary GA")
    #binary_ga(parameters, debug)

    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "minimization": True,
        "crossover_rate": 0.8,
        "mutation_rate": 0.1,
        "generations": 400,
        "tournament_total": 15,
        "population": 600,
        "sigma": 1
    }
    print("Executing Real-Valued GA")
    real_ga(parameters, debug)