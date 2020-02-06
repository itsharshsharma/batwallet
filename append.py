import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,4,12,8,13,6,10,2,22,30,17,4])
y = np.array([1.5,3,7,5,7.5,4,6,2,12,16,9.5,3])

m = np.size(y)
alpha = 0.01
t0, t1 = [], []

def y_pred(x, theta):
    return theta[0] + theta[1]*x

def cost_function(theta):
    cost = 1/(2*m)*np.sum((y_pred(x, theta) - y)**2)
    return cost

theta = [1,7]

def gradient_descent(m,alpha, theta, itteration):
    for i in range(itteration):
        y_pred = theta[0] + theta[1]*x
        d0 = 1/m*np.sum(y_pred - y)
        d1 = 1/m*np.sum((y_pred - y)*x)
        temp0 = theta[0] - 1/m*alpha*(d0)
        temp1 = theta[1] - 1/m*alpha*(d1)
        theta[0], theta[1] =  temp0, temp1
        t0.append(theta[0])
        t1.append(theta[1])
        plt.plot(i, cost_function(theta), 'r+')
        plt.plot()
    return theta[0], theta[1]

theta[0], theta[1] = gradient_descent(m, alpha, theta, 100)
print(theta)


t = y_pred(x, theta)
print(t)

plt.plot(x, y, c = 'y')
plt.plot(x, t, c = 'b')
plt.show()