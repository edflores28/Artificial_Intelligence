import sys
import csv
import numpy as np
from collections import Counter
from copy import deepcopy
from random import shuffle

class Node:
    '''
    Node class which represents the decision tree
    '''
    def __init__(self, feature):
        '''
        Initializes the object
        '''
        self.feature = feature
        self.children = {}

    def set_child(self, branch, node):
        '''
        Sets the branch and its value
        '''
        self.children[branch] = node

    def get_feature(self):
        '''
        Returns the feature value
        '''
        return self.feature

    def get_children(self):
        '''
        Returns the child nodes
        '''
        return self.children

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

def entropy(feature):
    '''
    This routine calculates the entropy of the feature

    Args:
        feature - the feature 
    Returns:
        the value for entropy
    '''
    # Obtain the unique counts of the feature set
    unique, counts = np.unique(feature, return_counts=True)
    # Set the probabilities accordingly
    p = counts / sum(counts)
    # Return the value for entropy
    return np.sum(np.negative(p)*np.log2(p))

def gain(s_entropy, label, feature):
    '''
    This routine calculates the gain of a feature

    Args:
        s_entropy - the entropy of the data
        label - the class labels
        feature - the feature
    Returns
        the value of the gain
    '''
    values = np.unique(feature)
    g = s_entropy
    for val in values:
        total = label[feature == val]
        g -= entropy(total)*(len(total)/len(feature))
    return g

def pick_best_feature(data, features):
    '''
    This routine determines the best feature

    Args:
        data - the data
        features - the features list
    Returns:
        the value of best feature
    '''
    # Transpose the data
    data = data.T
    # Index 0 contains the class values, calculate
    # the entropy
    set_entropy = entropy(data[0])
    # Calculate all the gains for the given features
    gains = [gain(set_entropy, data[0], data[x]) for x in features]
    # Transpose data
    data = data.T
    # Return the best feature
    return features[gains.index(max(gains))]

def is_homogeneous(data):
    '''
    This routine determines if the data
    is homogeneous

    Args:
        data - the data
    Returns:
        a boolean value
    '''
    if entropy(data.T[0]) == 0.0:
        return True
    else:
        return False

def get_majority_label(data):
    '''
    This routine gets the majority label
    of the data

    Args:
        data - the data
    Returns:
        the label
    '''
    # Obtain the labels and counts
    labels, counts = np.unique(data.T[0], return_counts=True)
    # Return the label with the maximum counts
    return labels[np.argmax(counts)]

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

def id3(data, features, default):
    '''
    This routine performs the ID3 algorithm

    Args:
        data - the data
        features - the features list
        default - the default label
    Returns
        a node
    '''
    # When the data is homogenous return the class label
    if is_homogeneous(data):
        return data[0][0]
    # When the features list is empty return
    # the majority class label
    if not features:
        return get_majority_label(data)
    # When the data list contains just the y values
    # return the default
    if len(data.T) == 1:
        return default
    # Determine the best feature
    best = pick_best_feature(data, features)
    # Create a node
    node = Node(best) 
    # Remove the best feature from the list    
    features.pop(features.index(best))
    # Determine the default label
    default_label = get_majority_label(data)
    # Iterate over the values and recursivly call id3()     
    for branch in np.unique(data.T[best]):
        idx = data[:, best] == branch
        subset = data[idx]
        child = id3(subset, features, default_label)
        node.set_child(branch, deepcopy(child))
    return node

def traverse(node, x):
    '''
    This routine traverses the tree using x

    Args:
        node - the node
        x - the test input
    '''
    children = node.get_children()
    value = x[node.get_feature()]
    try:
        # if the value is a string return it
        if isinstance(children[value], str):
            return children[value]
        # Otherwise continue traversing
        else:
            traverse(children[value], x)
    except:
        # If the key cannot be found return ?
        # since x cant be classified. The value
        # of the feature was not seen in the
        # training data
        return '?'

def train(training_data):
    """
    takes the training data and returns a decision tree data structure or ADT.

    Args:
        training_data - the data 
    Returns:
        a root node of a decision tree
    """
    return id3(training_data, [x for x in range(1, len(training_data[0]))], None)

def classify(tree, data):
    """
    takes the tree data structure/ADT and labeled/unlabeled data and 
    return a List of classifications.

    Args:
        tree - the decision tree
        data - the test data
    Returns:
        a list of classification
    """
    classifications = []
    for x in data:
        classifications.append(traverse(tree, x))
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
    # Iterate over the actual and predicted
    # values and count the errors
    for index in range(length):
        if actual[index] != predicted[index]:
            error += 1
    return error*100/length

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
        root_node = train(train_set)
        # Classify the test set
        classifications = classify(root_node, test_set)
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