from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    neural_network = NeuralNetwork(1, 3, 1)
    # print(neural_network.predict([1]))
    print(neural_network.learn([1], 1))
