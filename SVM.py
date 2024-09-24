import numpy as np
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, lambda_=0.01, epochs=1000):
        self.lambda_ = lambda_
        self.epochs = epochs
        self.w = None
        self.b = None

    # more