import numpy as np


class NeuralNetwork():
    def __init__(self, input_count, output_count, seed):
        '''
        constructor for neural network
        '''
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(seed)

        # We model our weight matrix based on input counts and output counts.
        # Then we assign random weights to it, with values in the range -1 to 1
        # and mean 0.
        self.input_count = input_count
        self.output_count = output_count
        self.__generate_rnd_weights()

    def __generate_rnd_weights(self):
        self.synaptic_weights = 2 * np.random.random((self.input_count, self.output_count)) - 1

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
        '''
        trains the network with a specified number of iterations
        returns the mean of errors generated by each round of training
        '''
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

    def train_until_fit(self, training_set_inputs, training_set_outputs, error_delta):
        '''
        trains the network until it is fit
        fit is defined as the difference between the errors generated
        in this iteration and last iteration is smaller or equal to
        the error_delta
        '''
        # how many times has training occurred
        count = 0;

        # error from last time
        last_error_mean = 0

        while (True):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            error_mean = np.mean(error)
            if (abs(error_mean - last_error_mean) < error_delta):
                print('Training is complete!');
                print('Training took {0} iterations to get fit'.format(count))
                break

            adjustments = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights += adjustments

            # store this error into the last error
            last_error_mean = error_mean
            count += 1

        return count

    def untrain(self):
        '''
        untrains the neural network
        '''
        self.__generate_rnd_weights()

    def predict(self, inputs):
        '''
        The neural network predicts an input
        '''
        # Pass inputs through our neural network (our single neuron).
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


