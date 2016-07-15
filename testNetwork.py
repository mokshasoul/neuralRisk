import numpy as np
import csv


class Neural_Network(object):
    def __init__(self, input, hidden, output):
        """
        :param input : number of input layers
        :param hidden: number of neuros in hidden layer
        :param output: number of neuronos in output layer
        """
        self.input = input + 1  # we add +1 for the bias
        self.hidden = hidden
        self.output = output
        # set up array of 1s for activations
        self.ai = [1.0] * self.input
        self.ah = [1.0] * self.hidden
        self.ao = [1.0] * self.output
        # create randomized weights
        self.wi = np.random.randn(self.input, self.hidden)
        self.wo = np.random.randn(self.hidden, self.output)
        # create array of 0 for backpropagation
        self.ci = np.zeros((self.input, self.hidden))
        self.co = np.zeros((self.hidden, self.output))

    def sigmoid(x):
        return (1/1 + np.exp(x))

    def dsigmoid(y):
        return y * (1.0 - y)

    def feedForward(self, inputs):
        if len(inputs) != self.input-1:

            raise ValueError('Wrong number of input vectors silly!!')
        # input activations
        for i in range(self.input - 1):  # - 1 to disregard the bias
            self.ai[i] = inputs[i]
        # hidden activation setup
        for j in range(self.hidden):
            sum = 0.0
            for i in range(self.input):
                sum += self.ai[i] * self.wi[i][j]
            self.ah[j] = self.sigmoid(sum)
        # output activation

        for k in range(self.output):
            sum = 0.0
            for j in range(self.hidden):
                sum += self.ah[j] * self.wo[j][k]
            self.ao[k] = self.sigmoid(sum)

        return self.ao[:]

    def backPropagate(self, targets, N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        """
        if len(targets) != self.outputs:
            raise ValueError("The number of targets is less than the output nodes \
                    specified")
            # calculate error term for output
        # the delta tell you which direction to change the weights (up or down)
        output_deltas = [0.0] * self.output
        for k in range(self.output):
            error = -(targets[k] - self.ao[k])
            output_deltas[k] = self.dsigmoid(self.ao[k]) * error
        # error terms for hidden nodes
        # delta tells you the amount of the error and in which direction to
        # change the weight (up or down)
        hidden_deltas = [0.0] * self.hidden
        for j in range(self.hidden):
            error = 0.0
            for k in range(self.output):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error
        # update the weights connecting hidden to output
        for i in range(self.hidden):
            for k in range(self.output):
                change = output_deltas[k] * self.ah[j]
                self.wi[i][j] -= N * change + self.ai[i]
                self.ci[i][j] = change
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def train(self, patterns, iterations=3000, N=0.02):
        # N : learning rate
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                error = self.backPropagate(targets, N)
            if i % 500 == 0:
                print('error %-.5f' % error)

    def predict(self, X):
        """
        return list of predictions after training algorithm
        """
        predictions = []
        for p in X:
            predictions.append(self.feedForward(p))
        return predictions
