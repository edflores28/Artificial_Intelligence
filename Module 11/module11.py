import sys
import csv


##
## I'm leaving the shape of your data to you.
## You may want to stick with our List of Lists format or
## you may want to change to a List of Dicts.
##
def read_data():
    pass

def train(training_data):
    """
    takes the training data and returns a decision tree data structure or ADT.
    """
    pass


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
    pass

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data()
    cross_validate(data)