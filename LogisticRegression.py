import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression():

    def __init__(self, X, y, alpha=0.01, lambda_=0.01):
        self.X: np.ndarray = X
        self.y: np.ndarray = y
        self.alpha = alpha
        self.lambda_ = lambda_
        self.W = np.array([np.random.rand() for i in range(X.shape[1])])
        # self.b = np.random.rand(1)

    def normalize(self):
        min = np.min(self.X)
        max = np.max(self.X)
        self.X = (self.X - min) / (max - min)

    def H(self, X: np.ndarray, W: np.ndarray):
        # y = w_i * x_i + b
        return np.dot(X, W.T)
    
    def s(self, X: np.ndarray):
        return 1 / (1 + np.exp(- self.H(X, self.W)))
    
    def Loss(self, y_aprox: np.ndarray):
        n = len(self.X)
        err = 1e-5
        y_aprox = np.clip(y_aprox, err, 1-err)
        reg = (self.lambda_/(2*n)) * np.sum(self.W ** 2)
        return (1/n) * -np.sum(self.y * np.log(y_aprox) + (1 - self.y) * np.log(1 - y_aprox)) + reg
    
    def Derivatives(self) -> np.ndarray:
        n = len(self.y)
        reg =2*(self.lambda_/n) * self.W 
        dw = (1/n) * np.dot((self.s(self.X) - self.y).T, self.X) + reg
        return dw
    
    def update(self, deriv: np.ndarray):
        self.W = self.W - self.alpha * deriv

    def train(self, epochs):
        self.normalize()
        loss = []
        y_aprox = self.s(self.X)
        for i in range(epochs):
            y_aprox = self.s(self.X)
            L = self.Loss(y_aprox)
            deriv = self.Derivatives()
            self.update(deriv)
            loss.append(L)
            # if ( (i % 1000) == 0):
            #   print("loss value error :" + str(L))
        return self.W, loss
    
    def reset(self):
        self.W = np.array([np.random.rand() for _ in range(self.X.shape[1])])

    def test(self, X):
        probs = self.s(X)
        y_pred = [1 if i > 0.5 else 0 for i in probs]
        return y_pred

    def Plot_Loss(self, epochs, loss):
        plt.plot(range(epochs), loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()