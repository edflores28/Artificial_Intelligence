import sys
import csv
import numpy as np
from collections import Counter
from copy import deepcopy
from random import shuffle

def read_data(filename):
    '''
    This routine reads the data from the file

    Args:
        filename - the files name
    Returns
        the data set as a numpy array
    '''
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([v for v in row])
    return np.asarray(data)

def determine_probability(feature, classes):
    '''
    This routine determines the probabilities
    of the feature

    Args:
        feature - a dict containing value counts
        classes - a dict contraining label counts
    Returns:
        a list of [P(p), P(e)]
    '''
    def calculate(feature, value, classes):
        '''
        The routine calculates the probability

        Args:
            feature - a dict containing value counts
            classes - a dict contraining label counts
            value - a key for the dicts
        Returns:
            the probability       
        '''
        # Calculate the probability with smoothing
        try:
            return (feature[value]+1)/(classes[value]+1)
        # When the key is not present in the feature 
        # dict apply just the smoothing
        except:
            return 1/(classes[value]+1)
    # Calculate P(p) and P(e)
    p = calculate(feature, 'p', classes)
    e = calculate(feature, 'e', classes)
    # Return the values
    return [p, e]

def generate_table(feature, data, class_counts):
    '''
    This routine generates a probability table for
    the given feature
    
    Args:
        feature - the index of the feature
        data - the data
        class_counts - the dict containing labels and counts
    Returns:
        a dict in the form of {value: [P(p), P(e)], ..}
    '''
    probs = {}
    # Only keep the class and feature column of the
    # data set
    f_data = data[:, [0, feature]]
    # Get the values and counts of the feature column
    values, counts = np.unique(f_data.T[1], return_counts=True)
    # Iterate over the values and create a subset of the
    # data that only contains the value and generate a 
    # probability table
    for val in range(len(values)):
        # Create a subset of the data which only contains
        # the val
        subset = f_data[f_data[:, 1] == values[val]]
        # Get the values and counts for the class labels
        sub_vals, sub_counts = np.unique(subset.T[0], return_counts=True)
        # Determine the probability of val and class labels
        probs[values[val]] = determine_probability(dict(zip(sub_vals, sub_counts)), class_counts)
    # Return the dict
    return probs

def calculate_probability(x, class_value, probs):
    '''
    This routine determines the probabily for
    x and the class value
    
    Args:
        x - the test input
        class_value - the label of the class
        probs - the probability table
    Returns:
        the probability
    '''
    class_prob = 1
    class_index = 0 if class_value == 'p' else 1
    # Iterate over the features minus the class label
    # and update class_prob
    for feature in range(1, len(x)):
        class_prob *= probs[feature][x[feature]][class_index]
    # Finally multiply the class label probability
    class_prob *= probs[0][class_value] / sum(probs[0].values())
    # Return the value
    return class_prob
    
def train(training_data):
    """
    takes the training data and returns the probabilities need for NBC.

    Args:
        training_data - the training data
    Returns:
        a dict of probabilities ins the form of:
        probs[feature] = {feature: {'value': [P(p), P(e)], ..}, ..}
    """
    prob = {}
    # Get the class labels and their counts
    values, counts = np.unique(training_data.T[0], return_counts=True)
    # Set key 0 with a dict of the label values and counts
    prob[0] = dict(zip(values, counts))
    # Iterate over all the features excluding the class labels and
    for feature in range(1, len(training_data[0])):
        # Generate a table for the feature and set the dict
        prob[feature] = generate_table(feature, training_data, prob[0])
    # Return the probability table
    return prob

def classify(probabilities, data):
    """
    Takes the probabilities needed for NBC, applies them to the data, and
    return a List of classifications.
    """
    classifications = []
    # Iterate over the data
    for x in data:
        # Calculate the values for p and e
        p = calculate_probability(x, 'p', probabilities)
        e = calculate_probability(x, 'e', probabilities)
        total = p + e
        # Add the dict to the list
        classifications.append({'e': e/total, 'p': p/total})
    # Return the list
    return classifications
    
def evaluate(actual, predicted):
    """
    takes a List of actual labels and a List of predicted labels
    and returns the error rate.
    """
    pass

def cross_validate(data):
    ##
    ## You can use your function from last week.
    ##
    ## combine train, classify and evaluate
    ## to perform 10 fold cross validation, print out the error rate for
    ## each fold and print the final, average error rate.
    probabilities = train(data)
    test = data[:10]
    classifications = classify(probabilities, test)
    for x in classifications:
        print(x)
    for x in test:
        print(x)
        
if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data("agaricus-lepiota.data")
    cross_validate(data)