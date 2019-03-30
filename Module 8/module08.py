import sys
import csv
import numpy as np
from copy import deepcopy

def read_data(filename):
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        data = []
        for row in csv_reader:
            data.append([float(v) for v in row])
    return data

def split_data(data):
    '''
    This routine takes data from the read_data routine
    converts it over to numpy arrays and splits
    the data into x and y. Also a 1's column is
    added to x

    Args:
        data - the data from the csv file
    Returns:
        x and y numpy arrays
    '''
    np_array = np.array(deepcopy(data))
    x = np_array[:, :-1]
    return np.hstack((np.ones((len(x), 1)), x)), np_array[:, -1]

def calculate_y_hat(thetas, x):
    '''
    This routine calculates the predicted
    y values for the given thetas and xs
    
    Args:
        thetas - the thetas
        x - the x values of the data
    Returns:
        the predicted y values
    '''
    return np.dot(x, thetas)

def calculcate_log_y_hat(thetas, x):
    '''
    This routine calculates the predicted
    y values for the given thetas and xs
    for logistic regression

    Args:
        thetas - the thetas
        x - the x values of the data
    Returns:
        the predicted y values
    '''
    values = np.negative(calculate_y_hat(thetas, x))
    return 1/(1+np.exp(values))

def calculate_error(y, y_hat):
    '''
    This routine calculates the error between
    y_hat - y. Used for linear regression

    Args:
        y_hat - the predicted y values
        y - the actual y values
    Returns:
        the error
    '''
    return np.sum(np.square((y_hat-y)))/2*len(y)

def calculate_log_error(y, y_hat):
    '''
    This routine calculates the error between
    y_hat - y. Used for logistic regression

    Args:
        y_hat - the predicted y values
        y - the actual y values
    Returns:
        the error
    '''
    return np.sum((y*np.log(y_hat))+((1-y)*np.log(1-y_hat)))/np.negative(len(y))

def calculate_derivative(x, y, y_hat, theta_idx):
    '''
    This routine calculates the derivative for both
    linear and logistic functions

    Args:
        x - the x values
        y - the y values
        y_hat - the predicted y values
        theta_index - The index of the theta value
    Returns:
        The derivative value
    '''
    return np.dot((y_hat-y), x.T[theta_idx])/len(y)

def gradient_descent(data, calc_y_hat, calc_error, debug, alpha, epsilon=1e-7):
    '''
    This routine performs gradient descent

    Args:
        data - the dataset
        calc_y_hat - routine that calculates the y_hats
        calc_error - routine that calculates the error
                     between y_hat and y
        debug - the debug flag
        alpha - the learning rate
        epsilon - the value for epsilon
    '''
    # Split the data into x and y 
    x, y = split_data(data)
    # Randomly generate theta valus for each x,
    # the valu 1 for theta_0 is accounted for
    thetas = np.random.uniform(-1.0, 1.0, len(x[0]))
    # Calculate the y_hats with the current
    # values of theta
    y_hat = calc_y_hat(thetas, x)
    previous_error = 0.0
    # Determine the current error of predicted
    # and actual y values
    current_error = calc_error(y, y_hat)
    new_thetas = np.array([0.0 for theta in range(len(thetas))])
    # Do gradient descent until current_error - previous_error
    # is less than epsilon
    while abs(current_error - previous_error) > epsilon:
        # Print the current error if the debug flag is set
        if debug:
            print("Current error:", current_error)
        # Update each theta
        for theta in range(len(new_thetas)):
            new_thetas[theta] = thetas[theta] - (alpha * calculate_derivative(x, y, y_hat, theta))
        thetas = new_thetas
        previous_error = current_error
        # Calculate the y_hats with the updated thetas
        y_hat = calc_y_hat(thetas, x)
        # Calculate the current error with the updated
        # y_hats
        current_error = calc_error(y, y_hat)
    # Return the thetas
    return thetas.tolist()

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
    alpha=0.01
    return gradient_descent(data, calculate_y_hat,calculate_error, debug, alpha)

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
    alpha=0.05
    return gradient_descent(data, calculcate_log_y_hat, calculate_log_error, debug, alpha)

def apply_linear_regression(model, xs):
    """
    model is the parameters of a linear regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    return np.dot(np.array([1.0] + xs), model)

def apply_logistic_regression(model, xs):
    """
    model is the parameters of a logistic regression model and the xs are the inputs
    to the model not including x_0 = 1.

    returns the predicted y based on the model and xs.
    """
    value = np.negative(np.dot(np.array([1.0] + xs), model))
    if 1 / (1 + np.exp(value)) > 0.5:
        return 1.0
    else:
        return 0.0

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    data = read_data("linear_regression.csv")
    linear_regression_model = learn_linear_regression(data, debug)
    print("linear regression model: ", linear_regression_model)
    for point in data[0:10]:
        print(point[-1], apply_linear_regression(linear_regression_model, point[:-1]))
    print()
    data = read_data("logistic_regression.csv")
    logistic_regression_model = learn_logistic_regression(data, debug)
    print("logistic regression model: ", logistic_regression_model)
    for point in data[0:10]:
        print(point[-1], apply_logistic_regression(logistic_regression_model, point[:-1]))