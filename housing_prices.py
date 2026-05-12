import pandas as pd
import numpy as np
import linear_regression_parameters as lrp
import cost_function as cost
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# Load CSV file
df = pd.read_csv("Univariate Linear Regression\\gdp-vs-happiness.csv")

# Select columns
x_train = df['GDP per capita'].to_numpy()
y_train = df['Life satisfaction'].astype(float).to_numpy()
# x_train = np.array([2, 4, 6, 8, 10])
# y_train = np.array([5, 9, 12, 15, 20])

slope = lrp.get_slope(x_train,y_train)
intercept = lrp.get_intercept(slope,x_train,y_train)
cost = cost.calculate_cost(x_train,y_train,slope,intercept)
print(f"Cost function evaluates to {cost:.2f}")

y_hat = lrp.calculate_predicted_values(slope, intercept,x_train,y_train)

print(f"Best fit line equation is : y(hat) = {intercept:.1f} + {slope:.8f} x_i")

print("Predict the happiness index of country 'Cyprus' having a GDP per capita of 37655")
x_test = 37655
prediction=intercept+(slope) * x_test
print("Happiness Index is" , prediction)

plt.scatter(x_train, y_train, marker="X", c="r", label='Actual Values')
plt.plot(x_train, y_hat, marker="o", c="b", label='Predicted Values')
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("Simple Linear Regression Model")
plt.legend()
plt.show()