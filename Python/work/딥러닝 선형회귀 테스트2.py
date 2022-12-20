
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
x_train = diabetes.data[0:3]
y_train = diabetes.target[:3]

raw_data = pd.read_csv('E:/GoogleDrive/A5팀 프로젝트 자료(12월22일)/dataset/product.csv',encoding='cp949')
x_train = np.array(raw_data.제품_적절성, dtype=np.float64)
y_train = np.array(raw_data.제품_만족도, dtype=np.float64)

class Neuron:
    def __init__(self):
        self.w = 1
        self.b = 1

    def forpass(self, x):
        y_hat = x * self.w + self.b
        return y_hat

    def backprop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def fit(self, x, y, epochs=1000):
        for i in range(epochs):
            for x_i, y_i in zip(x, y):
                y_hat = self.forpass(x_i)
                err = -(y_i - y_hat)
                w_grad, b_grad = self.backprop(x_i, err)
                self.w -= w_grad
                self.b -= b_grad
            if i % 100 == 0:
                print('[',i,']',err, self.w, self.b)



neuron = Neuron()
neuron.fit(x_train, y_train)
# print(neuron.w, neuron.b)

