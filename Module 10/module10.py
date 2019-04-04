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
    values = np.negative(np.dot(a, b))
    return 1/(1+np.exp(values))

def calculate_output_deltas(y_hat, y):
    return y_hat*(1-y_hat)*(y-y_hat)

def calculate_hidden_deltas(y_hat, output_deltas, output_thetas):
    sums = np.dot(np.transpose(output_thetas), output_deltas)
    return np.multiply(y_hat*(1-y_hat), sums)

def update_thetas(thetas, delta, inputs, alpha):
    return thetas + (alpha * delta * inputs)

def print_error(y_hat, y, debug):
    if debug:
        print(np.sum(np.square(y_hat-y)))
def learn_model(data, n_hidden, debug=False):
    # Split the data into x and y.
    # The x values will have the bias value
    # x[0]
    x, y = split_data(data)
    print(len(x), len(x[0]), len(y), len(y[0]))
    # Create the hidden and output layers with
    # thetas initialized between 0 and 1.
    hidden_thetas = np.array([np.random.random((1,len(x[0])))[0] for node in range(n_hidden+1)])
    hidden_layer = np.array([0.0 for node in range(n_hidden+1)])
    output_thetas = np.array([np.random.random((1,len(hidden_layer)))[0] for node in range(len(y[0]))])
    output_layer = np.array([0.0 for node in range(len(y[0]))])
    # Set the bias value at hidden_layer[0]
    hidden_layer[0] = 1.0
    # The learning rate
    alpha = 0.01
    for i in range(100):
        for index in range(len(x)):
            # Calculate the value for the hidden nodes
            # excluding hidden_layer[0]
            for node in range(1, len(hidden_layer)):
                hidden_layer[node] = calculate_output(hidden_thetas[node], x[index])
            # Calculate the value for the output nodes
            for node in range(len(output_layer)):
                output_layer[node] = calculate_output(output_thetas[node], hidden_layer)
            # Calculate all the deltas
            output_deltas = calculate_output_deltas(output_layer, y[index])
            hidden_deltas = calculate_hidden_deltas(hidden_layer, output_layer, output_thetas)
            # Iterate over the hidden and output layer thetas
            # and update them
            for theta in range(len(hidden_thetas)):
                hidden_thetas[theta] = update_thetas(hidden_thetas[theta], hidden_deltas[theta], x[0], alpha)
            for theta in range(len(output_thetas)):
                output_thetas[theta] = update_thetas(output_thetas[theta], output_deltas[theta], hidden_layer, alpha)
            print_error(output_layer, y[index], True)
    return [],[]

def apply_model(model, data, labeled=False):
    return []

def evaluate_results(results):
    return 1.00 # 100% error rate

if __name__ == "__main__":
    debug = len(sys.argv) > 1 and sys.argv[1].lower() == 'debug'

    train_data = generate_data(clean_data, 100)
    model = learn_model(train_data, 4, debug)

    # test_data = generate_data(clean_data, 100)
    # results = apply_model(model, test_data)

    # for result in results[0:10]:
    #     print(result)

    # error_rate = evaluate_results(results)
    # print(f"The error rate is {error_rate}")
