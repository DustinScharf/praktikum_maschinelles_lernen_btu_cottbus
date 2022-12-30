from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    neural_network = NeuralNetwork(1, 3, 1)
    print(neural_network.predict([1]))
    print(neural_network.learn([0, 0.349, 0.698, 3.141], [1, 0.939, 0.766, -1]))
