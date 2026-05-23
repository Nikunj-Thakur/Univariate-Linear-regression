import numpy as np
import utility_functions as uf
import pandas as pd

# # Load our data set
# x_train = np.array([1.0, 2.0])   #features
# y_train = np.array([300.0, 500.0])   #target value


df = pd.read_csv("Univariate Linear Regression\\gdp-vs-happiness.csv")

x_train = df['GDP per capita'].to_numpy()
y_train = df['Life satisfaction'].astype(float).to_numpy()

# Normalize features to prevent overflow
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std


# initialize parameters
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = uf.gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final:0.3f},{b_final:0.3f})")
