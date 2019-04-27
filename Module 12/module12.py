import sys
import csv
import numpy as np
from collections import Counter
from copy import deepcopy
from random import shuffle

def create_folds(data, k):
    '''
    This routine reads the csv file and
    creates k folds of the data set

    Args:
        data - the data
        k - the total folds
    Returns:
        a list of the data partitioned into folds
    '''
    # Shuffle the data a few times in order
    # to randomize the set
    shuffle(data)
    shuffle(data)
    folds = []
    # Calculate the length for each fold
    fold_len = int(len(data)/k)
    # Create k-1 folds
    for i in range(k-1):
        fold = []
        # Create the fold of size fold_len
        while len(fold) < fold_len:
            fold.append(data[0])
            data = np.delete(data, 0, 0)  
        # Add to the folds list
        folds.append(fold)
    # For the final fold add whatever remains in
    # the data set
    folds.append(data)
    # Return the folds
    return folds

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
        # Set val_count to the value of the key. If key is
        # not in the dict then set to 0
        val_count = feature[value] if value in feature.keys() else 0
        # Calculate the probability with smoothing
        return (val_count + 1)/(classes[value]+1)
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
    values = np.unique(f_data.T[1])
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
        try:
            class_prob *= probs[feature][x[feature]][class_index]
        except:
            # When a feature value is not in the dict then
            # just skip the feature. The +1 smoothing handles
            # these cases
            pass
    # Finally multiply the class label probability
    class_prob *= probs[0][class_value] / sum(probs[0].values())
    # Return the value
    return class_prob
    
def get_max_label(prob_dict):
    '''
    This routine reutns the label with the highest probability

    Args:
        prob_dict - dict with labels and probabilities
    Returns:
        a class label
    '''
    return max(zip(prob_dict.values(), prob_dict.keys()))[1]

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

     Args:
        actual - the real results
        predicted - the predicted results
    Returns:
        an error value   
    """
    error = 0
    length = len(actual)
    # Iterate over the actual and predicted values.
    for index in range(length):
        # Determine the label for the predicted dict and
        # increment the error count if not equal
        if get_max_label(predicted[index]) != actual[index]:
            error += 1
    # Return the error
    return (error*100)/length

def cross_validate(data):
    ''' 
    combine train, classify and evaluate
    to perform 10 fold cross validation, print out the error rate for
    each fold and print the final, average error rate.

    Args:
        data - the data set
    '''
    folds = create_folds(data, 10)
    errors = []
    print()
    # Iterate over the folds
    for fold in range(len(folds)):
        # Copy the entire data set
        train_set = deepcopy(folds)
        # Removing the fold from the training
        # list since this will be the testing set
        test_set = np.asarray(train_set.pop(fold))
        train_set = np.asarray([item for sublist in train_set for item in sublist])
        # Perform ID3 algorithm
        probabilities = train(train_set)
        # Classify the test set
        classifications = classify(probabilities, test_set)
        # Determine the error
        error = evaluate(test_set.T[0], classifications)
        errors.append(error)
        print("Fold", fold+1, "error", str(error)+"%")
    print()
    print("Average error", sum(errors)/len(folds))
    print()

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data("agaricus-lepiota.data")
    cross_validate(data)