import math
import numpy as np

'''
Vectorized version of Calculate Cost function. Vectorization is faster because it removes Python loops entirely
and executes operations in compiled C code with CPU-level optimizations [SIMD (Single Instruction, Multiple Data) + caching).
'''
def calculate_cost(x, y, b, w):
    m = len(x)
    prediction = w*x + b
    cost = np.sum((prediction-y)**2)
    return (1/(2*m)) * cost


'''
Vectorized version to Calculate Gradient for univariate linear regression model
'''
def calculate_gradient(x, y, b, w):
    m = x.shape[0]
    prediction = w*x + b
    dj_dw = (1/m) * np.sum((prediction-y)*x)
    dj_db = (1/m) * np.sum(prediction-y)
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, iterations):
     # An array to store cost J and prameters at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_init
    w = w_init

    for i in range(iterations):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = calculate_gradient(x, y, b, w)

        # Update Parameters simultaneously
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:   # prevent resource exhaustion 
            J_history.append(calculate_cost(x, y, b, w))
            p_history.append([w, b])

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(iterations/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:5.4f}",
                  f"dj/dw: {dj_dw:0.2f} dj/db: {dj_db:0.2f}",
                  f"w : {w:0.2f}  b :{b:0.2f}")

    return w, b, J_history, p_history


'''
Loop version of Calculate Cost function
For every single data point, Python is doing a lot of work.
Keeping it here as it helps to visulaize sometimes
'''
# def calculate_cost(x_train, y_train, b, w):
#     cost_total = 0
#     m = x_train.shape[0]
#     for i in range(m):
#         cost = ((x_train[i] * w + b) - y_train[i]) ** 2
#         cost_total = cost_total + cost
#     return (1/(2*m)) * cost_total


'''
Loop version of Calculate Gradient function for univariate linear regression model
For every single data point, Python is doing a lot of work.
'''
# def calculate_gradient(x, y, b, w):
#     # Number of training examples
#     m = x.shape[0]
#     dj_dw = 0
#     dj_db = 0

#     for i in range(m):
#         f_wb = w * x[i] + b
#         dj_dw_i = (f_wb - y[i]) * x[i]
#         dj_db_i = f_wb - y[i]
#         dj_db += dj_db_i
#         dj_dw += dj_dw_i
#     dj_dw = dj_dw / m
#     dj_db = dj_db / m
#     return dj_dw, dj_db


def mean(arr):
    sum = 0
    for ele in arr:
        sum = sum+ele
    return (sum/len(arr))


def mean_difference(arr, mean):
    diff = np.zeros(len(arr))
    for i in range(len(arr)):
        diff[i] = arr[i] - mean
    return diff


def muliply_mean_differences_and_sum(arrX, arrY):
    result = 0
    for i in range(len(arrX)):
        result = result + (arrX[i]*arrY[i])
    return result


def mean_diff_square_sum(arr):
    mean_sq = 0
    for i in range(len(arr)):
        mean_sq = mean_sq + (arr[i] * arr[i])
    return mean_sq


def get_slope(x_train, y_train):
    x_mean = mean(x_train)
    print("Mean of X is :", x_mean)
    y_mean = mean(y_train)
    print("Mean of Y is :", y_mean)
    x_mean_diff = mean_difference(x_train, x_mean)
    y_mean_diff = mean_difference(y_train, y_mean)
    x_y_mean_difference_product_sum = muliply_mean_differences_and_sum(
        x_mean_diff, y_mean_diff)
    x_mean_difference_square = mean_diff_square_sum(x_mean_diff)
    return x_y_mean_difference_product_sum/x_mean_difference_square


def get_intercept(slope, x_train, y_train):
    x_mean = mean(x_train)
    y_mean = mean(y_train)
    return y_mean-(slope)*x_mean


def calculate_predicted_values(slope, intercept, x_train, y_train):
    y_hat = np.zeros(len(x_train))
    for i in range(len(x_train)):
        y_hat[i] = intercept+(slope) * x_train[i]
    return y_hat