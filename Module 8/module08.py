import sys
import csv

def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([float(v) for v in row])
    return data

def learn_linear_regression(data, debug=False):
    """
    data is a list of lists where the last element. The outer list is all the data
    and each inner list is an observation or example for training. The last element of 
    each inner list is the target y to be learned. The remaining elements are the inputs,
    xs. The inner list does not include in x_0 = 1.

    This function uses gradient descent. If debug is True, then it will print out the
    the error as the function learns. The error should be steadily decreasing.

    returns the parameters of a linear regression model for the data.
    """
    pass


def learn_logistic_regression(data, debug=False):
    """
    data is a list of lists where the last element. The outer list is all the data
    and each inner list is an observation or example for training. The last element of 
    each inner list is the target y to be learned. The remaining elements are the inputs,
    xs. The inner list does not include in x_0 = 1.

    This function uses gradient descent. If debug is True, then it will print out the
    the error as the function learns. The error should be steadily decreasing.

    returns the parameters of a logistic regression model for the data.
    """
    pass

def apply_linear_regression(model, xs):
    """
    model is the parameters of a linear regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    pass

def apply_logistic_regression(model, xs):
    """
    model is the parameters of a logistic regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    pass

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'


    data = read_data("linear_regression.csv")
    linear_regression_model = learn_linear_regression(data, debug)
    print("linear regression model: ", linear_regression_model)
    for point in data[0:10]:
        print(point[-1], apply_linear_regression(linear_regression_model, point[:-1]))
    

    data = read_data("logistic_regression.csv")
    logistic_regression_model = learn_logistic_regression(data, debug)
    print("logistic regression model: ", logistic_regression_model)
    for point in data[0:10]:
        print(point[-1], apply_logistic_regression(logistic_regression_model, point[:-1]))