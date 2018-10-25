import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math

iris = load_iris()
newdata = iris.data
newdata = np.array(newdata)

#PCA降维
pca = PCA(n_components=2)
pca.fit(iris.data)
iris_1 = pca.transform(iris.data)

plt.figure(1)
plt.scatter(iris_1[:, 0], iris_1[:, 1])
plt.show()

#定义sigmoid函数
def my_sig(w, x):
    w = w.T
    r = np.dot(w, x)
    s = 1.0 / (1 + math.exp(-1*r))
    return s

#梯度下降法
def my_grad(w, x, y):
    len_y = len(y)
    sum_0 = 0
    for i in range(0, 99):
        s = my_sig(w, x[:, i])
        x1 = x[:, i]
        x1 = x1.reshape(x1.shape[0], 1)
        sum_0 = sum_0 + (s - y[i]) * x1

    g = 1 / len_y * sum_0
    return g

#损失函数
def my_fun(w, x, y):
    len_y = len(y)
    sum_0 = 0
    for i in range(0, len_y-1):
        s = my_sig(w, x[:, i])
        sum_0 = sum_0+y[i]*math.log(s)+(1-y[i])*math.log(1-s)
    f = -1/len_y*sum_0
    return f

#构造训练样本和测试样本
iris_1 = iris_1[:100, :]
iris_1 = iris_1.T
a = np.ones((1, 100))
x = np.vstack((iris_1, a))
y = np.vstack((np.ones((50, 1)), np.zeros((50, 1))))

#梯度下降法求解Logistic回归
w = np.array([[1], [1], [1]])
f_test = np.zeros(5000)
for i in range(0, 4999):
    w = w - 0.1 * my_grad(w, x, y)
    f_test[i] = my_fun(w, x, y)

plt.figure(2)
plt.plot(np.arange(5000), f_test)
plt.show()
y_test = np.zeros(100)
for i in range(0, 99):
    y_test[i] = my_sig(w, x[:, i])
print(y_test)
plt.figure(3)
plt.scatter(np.arange(100),y_test)
plt.show()