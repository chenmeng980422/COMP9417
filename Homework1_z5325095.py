#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/Users/chenmeng/Downloads/COMP9417/Homework/1/real_estate.csv')
df = pd.DataFrame(data)
df

df[df.isnull().values == True]

df = df.dropna()
df = df.drop(['transactiondate', 'latitude', 'longitude'], axis=1)

df['age_norm'] = (df.age - df.age.min()) / (df.age.max() - df.age.min())
df['nearestMRT_norm'] = (df.nearestMRT - df.nearestMRT.min()) / (df.nearestMRT.max() - df.nearestMRT.min())
df['nConvenience_norm'] = (df.nConvenience - df.nConvenience.min()) / (df.nConvenience.max() - df.nConvenience.min())
df['price_norm'] = (df.price - df.price.min()) / (df.price.max() - df.price.min())
df = df.drop(['age', 'nearestMRT', 'nConvenience', 'price'], axis=1)

df.mean()

data = np.array(df)
data_x = data[:, range(3)]
data_y = data[:, 3]
ones_data = np.ones([len(data), 1])
data_x_ones = np.column_stack((ones_data, data_x))

training_set = data[range(204), :]
ones = np.ones([len(training_set), 1])
train_x = training_set[:, range(3)]
train_y = training_set[:, 3]

test_set = data[range(204, 408), :]
ones = np.ones([len(training_set), 1])
test_x = test_set[:, range(3)]
test_y = test_set[:, 3]
test_x_ones = np.column_stack((ones, test_x))

# linearRegression
model = LinearRegression()
model.fit(train_x, train_y)
a = model.intercept_
b = model.coef_
predicted_y = model.predict(test_x)
print('Coefficients: ', a, b)
mse = np.mean((predicted_y - test_y) ** 2)
print("Mean squared error: %.2f" % mse)

# test


from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score


def performance_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    explained_var_score = explained_variance_score(y_true, y_pred)
    return rmse, r2, explained_var_score


rmsa, r2, explained_var_score = performance_metrics(test_y, predicted_y)
print("Root of mean squared error: %.2f" % rmsa)
print("R2-score: %.2f" % r2)
# explained_variance_score is shown in lecture alide 96 as R-squared: 
#  1-(variance(pred-target)/variance(pred-mean))
print("Explained variance score: %.2f" % explained_var_score)

ones = np.ones([len(train_x), 1])
train_x_ones = np.column_stack((ones, train_x))
theta_ini = np.ones([4, 1])


def return_y_estimate(theta_now, x):
    y_estimate = np.dot(x, theta_now)
    return y_estimate


def return_loss_function(theta_now, x, y):
    y_estimate = return_y_estimate(theta_now, x)
    n = x.shape[0]
    m = x.shape[1]
    sum = 0
    for i in range(n):
        sum += (0.25 * (y[i] - y_estimate[i]) ** 2 + 1) ** 0.5 - 1
    l = sum / n
    return l


def gradient_descent(x, y, learning_rate, theta_now):
    max_loop = 400
    tmp = np.zeros(theta_now.shape)
    n = x.shape[0]
    m = x.shape[1]
    y = y.reshape(-1, 1)
    cost = np.zeros(400)
    theta_list = [[], [], [], []]

    for i in range(max_loop):
        for j in range(m):
            theta_list[j].append(theta_now[j][0])
        err = return_y_estimate(theta_now, x) - y
        for j in range(m):
            gradient = np.dot(x[:, j].T, err) / (2 * (np.dot(err.T, err) + 4) ** 0.5) / n
            tmp[j, 0] = theta_now[j, 0] - (learning_rate * gradient)
        theta_now = tmp
        cost[i] = return_loss_function(theta_now, x, y)
    return theta_list, cost


losses = []
fig, ax = plt.subplots(3, 3, figsize=(10, 10))
alphas = [10, 5, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01]
for i, ax in enumerate(ax.flat):
    losses.append(gradient_descent(train_x_ones, train_y, alphas[i], theta_ini)[1])
    ax.plot(losses[i])
    ax.set_title(f"step size: {alphas[i]}")
plt.tight_layout()
plt.show()

fig1, ax1 = plt.subplots(2, 2, figsize=(6, 6))
thetas = gradient_descent(train_x_ones, train_y, 0.3, theta_ini)[0]
for i, ax1 in enumerate(ax1.flat):
    ax1.plot(thetas[i])
    ax1.set_title(f"theta{[i]}")
plt.tight_layout()
plt.show()

theta_final = [0, 0, 0, 0]
for i in range(4):
    theta_final[i] = thetas[i][-1]
print(theta_final)

theta_final = np.array(theta_final).reshape([4, 1])
theta_final

y_hat = return_y_estimate(theta_final, test_x_ones)
y_hat
achieved_loss = return_loss_function(theta_final, test_x_ones, test_y)
print("The achieved loss is: ", achieved_loss)


def stochastic_gradient_descent(x, y, learning_rate, theta_now):
    epochs = 6
    tmp = np.zeros(theta_now.shape)
    n = x.shape[0]
    m = x.shape[1]
    y = y.reshape(-1, 1)
    cost = []
    theta_list = [[], [], [], []]

    for i in range(epochs):
        for j in range(n):
            err = return_y_estimate(theta_now, x)[j] - y[j]
            for k in range(m):
                gradient = x[j][k] * err / (2 * (err ** 2 + 4) ** 0.5)
                tmp[k, 0] = theta_now[k, 0] - (learning_rate * gradient)
            theta_now = tmp
            cost.append(return_loss_function(theta_now, x, y)[0])
            for j in range(m):
                theta_list[j].append(theta_now[j][0])
    return theta_list, cost


losses = []
fig2, ax2 = plt.subplots(3, 3, figsize=(10, 10))
for i, ax2 in enumerate(ax2.flat):
    losses.append(stochastic_gradient_descent(train_x_ones, train_y, alphas[i], theta_ini)[1])
    ax2.plot(losses[i])
    ax2.set_title(f"step size: {alphas[i]}")
plt.tight_layout()
plt.savefig("q6a.png")
plt.show()

fig3, ax3 = plt.subplots(2, 2, figsize=(6, 6))
thetas_sgd = stochastic_gradient_descent(train_x_ones, train_y, 0.4, theta_ini)[0]
for i, ax3 in enumerate(ax3.flat):
    ax3.plot(thetas_sgd[i])
    ax3.set_title(f"theta{[i]}")
plt.tight_layout()
plt.savefig("q6c.png")
plt.show()

theta_final_sgd = [0, 0, 0, 0]
thetas_sgd = stochastic_gradient_descent(train_x_ones, train_y, 0.4, theta_ini)[0]
for i in range(4):
    theta_final_sgd[i] = thetas_sgd[i][-1]
theta_final_sgd = np.array(theta_final_sgd).reshape([4, 1])
theta_final_sgd

y_hat_sgd = return_y_estimate(theta_final_sgd, test_x_ones)

achieved_loss = return_loss_function(theta_final_sgd, test_x_ones, test_y)
print("The achieved loss is: ", achieved_loss)
