import numpy as np


class NeuralNetwork():
    def __init__(self, input_count, output_count, seed):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(seed)

        # We model our weight matrix based on input counts and output counts.
        # Then we assign random weights to it, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = 2 * np.random.random((input_count, output_count)) - 1

    def __sigmoid(self, x):
        '''
        The Sigmoid function, which describes an S shaped curve.
        We pass the weighted sum of the inputs through this function to
        normalise them between 0 and 1.
        '''
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        '''
        The derivative of the Sigmoid function.
        This is the gradient of the Sigmoid curve.
        It indicates how confident we are about the existing weight.
        '''
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        errors = []
        for iteration in range(number_of_training_iterations):
            '''
            We train the neural network through a process of trial and error
            adjusting the synaptic weights each time.
            '''
            # Pass the training set through our neural network.
            output = self.predict(training_set_inputs)

            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = training_set_outputs - output
            errors.append(np.mean(error))

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            self.synaptic_weights += adjustment

        return errors

    def predict(self, inputs):
        '''
        The neural network predicts an input
        '''
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


