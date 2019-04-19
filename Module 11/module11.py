import sys
import csv
import numpy as np
from collections import Counter
from copy import deepcopy
from random import shuffle

class Node:
    def __init__(self, feature):
        self.feature = feature
        self.children = {}

    # def set_feature(self, feature):
    #     self.feature = feature

    def set_child(self, branch, node):
        self.children[branch] = node

    def get_feature(self):
        return self.feature

    def get_children(self):
        return self.children

def create_folds(data, k):
    '''
    This routine reads the csv file and
    creates k folds of the data set
    '''
    shuffle(data)
    fold_len = int(len(data)/k)
    return [data[i:i+fold_len] for i in range(0, len(data), fold_len)]

def entropy(feature):
    '''
    This routine calculates the entropy of the feature
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
    '''
    values = np.unique(feature)
    g = s_entropy
    for val in values:
        total = label[feature == val]
        g -= entropy(total)*(len(total)/len(feature))
    return g

def pick_best_feature(data, features):
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
    if entropy(data.T[0]) == 0.0:
        return True
    else:
        return False

def get_majority_label(data):
    # Obtain the labels and counts
    labels, counts = np.unique(data.T[0], return_counts=True)
    # Return the label with the maximum counts
    return labels[np.argmax(counts)]

def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([v for v in row])
    return np.asarray(data)

def id3(data, features, default):
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
    children = node.get_children()
    value = x[node.get_feature()]
    if isinstance(children[value], str):
        return children[value]
    else:
        traverse(children[value], x)

def train(training_data):
    """
    takes the training data and returns a decision tree data structure or ADT.
    """
    return id3(training_data, [x for x in range(1, len(training_data[0]))], None)

def classify(tree, data):
    """
    takes the tree data structure/ADT and labeled/unlabeled data and 
    return a List of classifications.
    """
    pass


def evaluate(actual, predicted):
    """
    takes a List of actual labels and a List of predicted labels
    and returns the error rate.
    """
    pass

def cross_validate(data):
    ## combine train, classify and evaluate
    ## to perform 10 fold cross validation, print out the error rate for
    ## each fold and print the final, average error rate.
    folds = create_folds(data, 10)

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data("agaricus-lepiota.data")
    cross_validate(data)