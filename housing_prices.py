import pandas as pd
import numpy as np
import linear_regression_parameters as lrp
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Load CSV file
df = pd.read_csv("Univariate Linear Regression\\gdp-vs-happiness.csv")


def get_slope():
    x_mean = lrp.mean(x_train)
    y_mean = lrp.mean(y_train)
    x_mean_diff = lrp.mean_difference(x_train, x_mean)
    y_mean_diff = lrp.mean_difference(y_train, y_mean)
    x_y_mean_difference_product_sum = lrp.muliply_mean_differences_and_sum(
        x_mean_diff, y_mean_diff)
    x_mean_difference_square = lrp.mean_square(x_mean_diff)
    return x_y_mean_difference_product_sum/x_mean_difference_square


def get_intercept(slope):
    x_mean = lrp.mean(x_train)
    y_mean = lrp.mean(y_train)
    return y_mean-(slope)*x_mean


def calculate_predicted_values(slope, intercept):
    y_hat = np.zeros(len(x_train))
    for i in range(len(x_train)):
        y_hat[i] = intercept+(slope) * x_train[i]
    return y_hat


# Select columns
x_train = df['GDP per capita'].to_numpy()
y_train = df['Life satisfaction'].astype(float).to_numpy()
# x_train = np.array([2, 4, 6, 8, 10])
# y_train = np.array([5, 9, 12, 15, 20])

slope = get_slope()
intercept = get_intercept(slope)

y_hat = calculate_predicted_values(slope, intercept)

print(f"Best fit line equation is : y(hat) = {intercept:.1f} + {slope} x_i")

plt.scatter(x_train, y_train, marker="X", c="r", label='Actual Values')
plt.plot(x_train, y_hat, marker="o", c="b", label='Predicted Values')
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("Simple Linear Regression Model")
plt.legend()
plt.show()

print("Predict the happiness index of country 'Cyprus' having a GDP per capita of 37655")
x_test = 37655
prediction=intercept+(slope) * x_test
print("Happiness Index is" , prediction)