import numpy as np


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_input_to_hidden = np.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                        (self.hidden_nodes, self.input_nodes))
        self.weights_hidden_to_output = np.random.normal(0.0, pow(self.output_nodes, -0.5),
                                                         (self.output_nodes, self.hidden_nodes))

        self.bias_hidden = np.array([[1]] * self.hidden_nodes)
        # self.bias_output = np.array([[1]] * self.output_nodes)  # not a task, but experimental

    def activation_function(self, x):  # die sigmund funktion eigentlich
        return 1 / (1 + np.exp(-x))

    def predict(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.weights_input_to_hidden, inputs)
        hidden_inputs = np.add(hidden_inputs, self.bias_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.weights_hidden_to_output, hidden_outputs)
        # final_inputs = np.add(final_inputs, self.bias_output)  # not a task, but experimental
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def learn(self, train_data, train_labels):
        iterations = 100

        beta = 0.9

        theta_new = np.stack(
            (self.weights_hidden_to_output.T, self.bias_hidden, self.weights_input_to_hidden),
            axis=1
        ).flatten()

        theta_yet = theta_new
        theta_old = theta_yet  # todo fragen: oder 0 setzen?

        error_yet = np.inf  # todo maybe error

        alpha = 0.001

        for k, data in enumerate(train_data):
            result = self.predict(data)

            theta_new = theta_yet - alpha * ABLEITUNG + beta * DELTA * theta_old  # todo

            error_new = self.E(train_labels, result)

            # todo update weights according to theta_new

            alpha = alpha if np.less(error_new, error_yet) else alpha / 2

            theta_old = theta_yet
            theta_yet = theta_new

            error_yet = error_new

        def E(d, z):
            s = len(d)  # s ist länge der trainingsdaten
            return 1 / (2 * s) * np.sum(np.power(np.subtract(d, z), 2))
