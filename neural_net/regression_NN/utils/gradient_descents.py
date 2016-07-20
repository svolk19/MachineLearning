
def batch_gradient_descent(weights, biases, weight_gradients, bias_gradients, X, y, iterations=5000, learning_rate=0.01,
                           reg_lambda=0.01, regularize=False):
    """
    implements simple batch gradient descent
    :param weights: tuple of numpy weight arrays
    :param biases: tuple of numpy bias arrays
    :param weight_gradients: tuple of numpy arrays of weight gradients
    :param bias_gradients: tuple of numpy arrays of bias gradients
    :param X: input data
    :param y: output data
    :param learning_rate:
    :param reg_lambda: regularization rate
    :param print_accuracies: print accuracies at each iteration
    :return: new weights, new biases
    """

    for num in range(iterations):

        # iterate through weights and add regularization
        if regularize:
            for weight_scheme in weights:
                for weight in weight_scheme:
                    weight += reg_lambda * weight

        # iterate through weight gradients and adjust weights
        for i, weight_gradient_scheme in enumerate(weight_gradients):
            for j, weight_gradient in enumerate(weight_gradient_scheme):
                weights[i][j] -= learning_rate * weight_gradient

        # iterate through bias gradients and adjust biases
        for i, bias_gradient_scheme in enumerate(bias_gradients):
             for j, bias_gradient in enumerate(bias_gradient_scheme):
                biases[i][j] -= learning_rate * bias_gradient



    return weights, biases

# def mini_batch_gradient_descent():
#     # implementation of simple stochastic mini batch gradient descent
#
#


























