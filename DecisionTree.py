import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = None

    # more