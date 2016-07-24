def tan_normalize(number):
    # assumes accuracy a float between 0.0 and 1.0
    # returns a float -pi/2 < X < pi/2, normalized within the range of the tangent function

    number = number * np.pi
    
    if number < np.pi / 2:
        result = -number
    else:
        result = number - (np.pi / 2)
        
    return result

def assign_rate(X, y):
    # assign a learning rate based on neural network accuracy
    # new learning rate is based on the tangent of the current accuracy

    accuracy = self.accuracy(X, y)
    accuracy_normalized = self.tan_normalize(accuracy)
    self.learning_rate -= np.tan(accuracy_normalized)
