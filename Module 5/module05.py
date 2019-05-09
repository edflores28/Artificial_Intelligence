import sys
from random import choice
from random import randrange
from random import random
from random import sample
from copy import deepcopy
from random import gauss, random
from collections import deque
from operator import itemgetter
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

def generate_population(size, function, is_binary, n=10):
    '''
    This routine generates a random population

    Args:
        size - the total population size
        function - the shift sphere function
    Returns:
        the population with fitness scores
    '''
    # Generator to generate float values in specific range
    def float_range(start, stop, increment):
        start = Decimal(start)
        increment = Decimal(increment)
        while start < stop:
            yield float(start)
            start += increment
    population = []
    # Binary ga only uses values list of 0 and 1
    # Otherwise for real ga set values list
    # to -5.12 to 5.12   
    values = [0, 1] if is_binary else list(float_range(-5.12, 5.12, Decimal('0.01')))
    multiplier = n if is_binary else 1
    # Create the populate for the desired size and also
    # calculate the fitness scores
    while len(population) < size:
        individual = deepcopy([choice(values) for x in range(n*multiplier)])
        score = []
        if is_binary:
            score = fitness(convert_bin(individual), function)
        else:
            score = fitness(individual, function)
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
    # Get a sample of the total number of parents
    # and get the top 2 best values base on the fitness
    # score in index 0
    return tuple(sorted(sample(population, total), key=itemgetter(0))[-2:])
           
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
        #index = randrange(len(parent_1[1])) + 1
        index = randrange(1, len(parent_1[1])-2)
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
    # Iterate over the length
    for x in range(length):
        # Generate a random value and if less than
        # the rate/length mutate.
        if random() <= rate/length:
            # Flip the bit for binary individual
            if is_binary:
                individual[x] = 1 if individual[x] == 0 else 0
            # Otherwise add some noise to real values
            else:
                individual[x] += gauss(0.0, sigma)
                # Contrain the individual to the range
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

def fitness(individual, function):
    '''
    This routine calculates the fitness score

    Args:
        individual - the individual to calculate
        function - the shift sphere function
    Returns:
        The fitness score
    '''
    return 1 / (1 + function(individual))

def calculate_scores(child_1, child_2, function, is_binary):
    '''
    This routine calculates the fitness score of the offspring

    Args:
        child_1 - the first child
        child_2 - the second child
    Returns:
        returns the offspring with the fitness score
    '''
    if is_binary:
        fitness_1 = fitness(convert_bin(child_1), function)
        fitness_2 = fitness(convert_bin(child_2), function)
    else:
        fitness_1 = fitness(child_1, function)
        fitness_2 = fitness(child_2, function)
    return [fitness_1] + [child_1], [fitness_2] + [child_2]

def get_best_fit(population):
    '''
    This routine finds the best fit individual
    
    Args:
        population - the entire population
    Returns:
        best fit individual
    '''
    return sorted(deepcopy(population), key=itemgetter(0))[-1:][0]

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
    population = generate_population(parameters['population'], parameters['f'], is_binary)
    # Iterate for a total generations
    for generation in range(parameters['generations']):
        # Select two parents
        parent_1, parent_2 = select_parents(population, parameters['tournament_total'])
        # Create child from the parents
        child_1, child_2 = crossover(parent_1, parent_2, parameters['crossover_rate'])
        # Mutate the children
        child_1, child_2 = mutate(child_1, child_2, parameters['mutation_rate'], is_binary, parameters['sigma'])
        # Calculate the fitness scores for the children
        child_1, child_2, = calculate_scores(child_1, child_2, parameters['f'], is_binary)
        # Add the children back to the population
        population.append(child_1)
        population.append(child_2)
        if debug:
            print("Generation", generation + 1)
            print_indv(get_best_fit(population), parameters['f'], is_binary, debug)
    # Only print the best individual on the last generation
    # if debug is False
    if not debug:
        print("\nGenetic Algorithm Complete")
        print("Best Fit Individual:")
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
        "crossover_rate": 0.8,
        "mutation_rate": 0.05,
        "generations": 2000,
        "tournament_total": 15,
        "population": 500,
        "sigma": 0 # Not used for binary ga, still needs to be defined.
    }
    print("Executing Binary GA")
    binary_ga(parameters, debug)

    parameters = {
        "f": lambda xs: shifted_sphere( 0.5, xs),
        "crossover_rate": 0.85,
        "mutation_rate": 0.05,
        "generations": 2000,
        "tournament_total": 15,
        "population": 500,
        "sigma": 0.7
    }
    print("Executing Real-Valued GA")
    real_ga(parameters, debug)