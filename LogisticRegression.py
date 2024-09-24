import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lambda_=0.01, epochs=1000):
        self.lambda_ = lambda_
        self.epochs = epochs

    def H(self, z):
        return 1 / (1 + np.exp(-z))
    

    # more