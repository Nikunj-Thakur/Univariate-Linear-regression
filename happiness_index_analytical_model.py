import pandas as pd
import numpy as np
import utility_functions as uf
import utility_functions as cost
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = pd.read_csv("Univariate Linear Regression\\gdp-vs-happiness.csv")

x_train = df['GDP per capita'].to_numpy()
y_train = df['Life satisfaction'].astype(float).to_numpy()

# Scaling the feture using Z-score standardisation to prevent overflow
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std

# x_train = np.array([2, 4, 6, 8, 10])
# y_train = np.array([5, 9, 12, 15, 20])

slope = uf.get_slope(x_train, y_train)
print("Slope is :", slope)
intercept = uf.get_intercept(slope, x_train, y_train)
cost = cost.calculate_cost(x_train, y_train, intercept, slope)
print(f"Cost function evaluates to {cost:.2f}")

y_hat = uf.calculate_predicted_values(slope, intercept, x_train, y_train)

print(f"Best fit line equation is : y(hat) = {intercept:.3f} + {slope:.3f} x_i")

print("Predict the happiness index of country 'Cyprus' having a GDP per capita of 37655")
x_test = 37655
X_new_scaled = (x_test - x_mean) / x_std
prediction = intercept+(slope) * X_new_scaled
print("Happiness Index is", f"{prediction:0.2f}")

plt.scatter(x_train, y_train, marker="X", c="r", label='Actual Values')
plt.plot(x_train, y_hat, marker="o", c="b", label='Predicted Values')
plt.xlabel("GDP per capita, standardised values")
plt.ylabel("Life satisfaction")
plt.title("Simple Linear Regression Model")
plt.legend()
plt.show()
