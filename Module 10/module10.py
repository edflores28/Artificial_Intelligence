import sys
import random
import numpy as np
from copy import deepcopy

clean_data = {
    "plains": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]
    ],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"]
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"]
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"]        
    ]
}

def blur(data):
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v
    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]

def generate_data(clean_data, total_per_label):
    labels = len(clean_data.keys())
    def one_hot(n, i):
        result = [0.0] * n
        result[i] = 1.0
        return result

    data = []
    for i, label in enumerate(clean_data.keys()):
        for _ in range(total_per_label):
            datum = blur(random.choice(clean_data[label]))
            xs = datum[0:-1]
            ys = one_hot(labels, i)
            data.append((xs, ys))
    random.shuffle(data)
    return data

def split_data(data):
    '''
    This routine splits the data into x and y.
    The bias term is also added to the x values

    Args:
        data - the data stored as tuples ([x], [y])
    Returns:
        x and y numpy arrays
    '''
    x = []
    y = []
    for item in data:
        x.append([1.0]+deepcopy(item[0]))
        y.append(deepcopy(item[1]))
    return np.array(x), np.array(y)

def calculate_output(a, b):
    '''
    This routine calculates the node
    outputs

    Args:
        a - a numpy array
        b - a numpy array
    Return:
        A nodes output
    '''
    values = np.negative(np.dot(a, b))
    return 1/(1+np.exp(values))

def calculate_output_deltas(y_hat, y):
    '''
    This routine calculates the output
    nodes delta values

    Args:
        y_hat - the predicted y values
        y - the actual y values
    Returns:
        A numpy array containing all the output
        nodes deltas
    '''
    return y_hat*(1-y_hat)*(y-y_hat)

def calculate_hidden_deltas(y_hat, output_deltas, output_thetas):
    '''
    This routine calculates the hidden
    nodes delta values

    Args:
        y_hat - the predicted y values
        output_deltas - the output nodes delta values
        output_thetas - the output nodes theta values
    Returns:
        A numpy array containing all the hidden
        nodes deltas
    '''
    sums = np.dot(np.transpose(output_thetas), output_deltas)
    return y_hat*(1-y_hat) * sums

def update_thetas(thetas, delta, inputs, alpha):
    '''
    This routine updates the given thetas

    Args:
        thetas - the thetas to update
        delta - the delta values
        inputs - the inputs
        alpha - the learning rate
    Returns:
        The updated theta value
    '''
    return thetas + (alpha * delta * inputs)

def calculate_error(y_hat, y):
    '''
    This routin calculates the error

    Args:
        y_hat - the predicated y values
        y - the actual y values
    '''
    return np.negative(np.sum(y*np.log(y_hat)))

def feed_forward(hidden_thetas, output_thetas, x):
    '''
    This routines performs the feed forward portion
    of the neural network

    Args:
        hidden_thetas - the hidden layers thetas
        output_thetas - the output layers thetas
        x - the input values
    Returns:
        numpy arrays for the hidden and outer layers
        outputs
    '''
    hidden_outputs = np.array([0.0 for node in range(len(hidden_thetas))])
    outputs = np.array([0.0 for node in range(len(output_thetas))])
    # Calculate the outputs for the hidden nodes
    for node in range(len(hidden_outputs)):
        hidden_outputs[node] = calculate_output(hidden_thetas[node], x)
    # Add the bias value to the hidden outputs
    hidden_outputs = np.insert(hidden_outputs, 0, 1.0)
    # Calculate the output nodes values
    for node in range(len(outputs)):
        outputs[node] = calculate_output(output_thetas[node], hidden_outputs)
    return hidden_outputs, outputs

def learn_model(data, n_hidden, debug=False):
    # Split the data into x and y.
    # The x values will have the bias value
    # x[0]
    x, y = split_data(data)
    # Create the hidden and output layers with
    # thetas initialized between 0 and 1.
    hidden_thetas = np.array([np.random.random((1,len(x[0])))[0] for node in range(n_hidden)])
    output_thetas = np.array([np.random.random((1,n_hidden+1))[0] for node in range(len(y[0]))])
    # Set the bias value at hidden_layer[0]
    #hidden_layer[0] = 1.0
    # The learning rate
    alpha = 0.001
    # The value for epsilon
    epsilon = 1e-7
    previous_error = 0.0
    current_error = 0.0
    counter = 0
    while True:
        current_error = 0.0
        for index in range(len(x)):
            # Calculate the hidden and output layers y hats
            hidden_layer, output_layer = feed_forward(hidden_thetas, output_thetas, x[index])
            # Calculate all the deltas
            output_deltas = calculate_output_deltas(output_layer, y[index])
            hidden_deltas = calculate_hidden_deltas(hidden_layer, output_layer, output_thetas)
            # Iterate over the hidden and output layer thetas
            # and update them
            for theta in range(1, len(hidden_thetas)):
                hidden_thetas[theta] = update_thetas(hidden_thetas[theta], hidden_deltas[theta+1], x[index], alpha)
            for theta in range(len(output_thetas)):
                output_thetas[theta] = update_thetas(output_thetas[theta], output_deltas[theta], hidden_layer, alpha)
            current_error += calculate_error(output_layer, y[index])
        if abs(current_error - previous_error) < epsilon:
            break
        previous_error = current_error
        counter += 1
        if counter == 500 and debug:
            print("Current Error:", current_error)
            counter = 0
    return hidden_thetas.tolist(),output_thetas.tolist()

def apply_model(model, data, labeled=False):
    x, y = split_data(data)
    results = []
    for index in range(len(x)):
        hidden_layer, output_layer = feed_forward(model[0], model[1], x[index])
        if labeled:
            max_index = np.argmax(output_layer)
            converted = [0 for i in range(len(output_layer))]
            converted[max_index] = 1
            results.append([(y[index][i], int(converted[i])) for i in range(len(y[index]))])
        else:
            results.append([(y[index][i], output_layer[i]) for i in range(len(y[index]))])
    return results

def evaluate_results(results):
    correct = 0
    # Iterate over the results and see
    # if the (1, 1) tuple exists in the
    # result. This assumes that the
    # results are labeled
    for result in results:
        if (1, 1) in result:
            correct += 1
    return correct / len(results)

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    train_data = generate_data(clean_data, 100)
    model = learn_model(train_data, 4, debug)

    test_data = generate_data(clean_data, 100)
    results = apply_model(model, test_data, True)

    for result in results[0:10]:
        print(result)

    error_rate = evaluate_results(results)
    print(f"The error rate is {error_rate}")
