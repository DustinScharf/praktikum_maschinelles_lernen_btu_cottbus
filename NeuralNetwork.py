import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        np.random.seed(12345)
        self.weights_input_to_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), self.hidden_nodes)
        self.weights_hidden_to_output = np.random.normal(0.0, pow(self.output_nodes, -0.5), self.hidden_nodes)

        # self.bias_hidden = np.array([[1]] * self.hidden_nodes)
        self.bias_hidden = np.ones(self.hidden_nodes, dtype=np.float64)
        # self.bias_output = np.array([[1]] * self.output_nodes)  # not a task, but experimental

    def activation_function(self, x):  # die sigmund funktion eigentlich
        return 1 / (1 + np.exp(-x))

    def predict(self, input_list):
        inputs = np.array(input_list)
        # inputs = np.array(input_list, ndmin=2)

        hidden_inputs = np.multiply(self.weights_input_to_hidden, inputs)
        hidden_inputs = np.add(hidden_inputs, self.bias_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # final_inputs = np.add(final_inputs, self.bias_output)  # not a task, but experimental
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def learn(self, train_data, train_labels):
        iterations = 100

        beta = 0.9

        theta_new = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        theta_yet = self.weights_input_to_hidden, self.bias_hidden, self.weights_hidden_to_output
        theta_old = 0, 0, 0
        delta_theta_old = [0, 0, 0], [0, 0, 0], [0, 0, 0]

        error_yet = np.inf  # todo maybe error

        alpha = 0.001

        def err(d, z):
            # s = len(d)  # s ist länge der trainingsdaten
            s = 1
            return 1 / (2 * s) * np.sum(np.power(np.subtract(d, z), 2))

        for k, data in enumerate(train_data):
            result = self.predict(data)

            def theta_update(index):
                return theta_yet[index] - alpha * derivation + np.multiply(beta, delta_theta_old[index])

            sigmoid_derivation = ((theta_yet[0] * data + theta_yet[1]) *
                                  (1 - (theta_yet[0] * data + theta_yet[1])))

            derivation = (result - train_labels[k]) * theta_yet[2] * data * sigmoid_derivation
            theta_new[0] = theta_update(0)

            derivation = (result - train_labels[k]) * theta_yet[2] * sigmoid_derivation
            theta_new[1] = theta_update(1)

            derivation = (result - train_labels[k]) * sigmoid_derivation
            theta_new[2] = theta_update(2)

            error_new = err(train_labels[k], result)

            # todo update weights according to theta_new (or is it already referenced?)

            alpha = alpha if np.less(error_new, error_yet) else alpha / 2

            delta_theta_old = -alpha * derivation + np.multiply(beta, delta_theta_old)
            # theta_old = theta_yet
            theta_yet = theta_new

            error_yet = error_new
