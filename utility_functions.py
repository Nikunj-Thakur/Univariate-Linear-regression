import math
import numpy as np

# Loop version of Calculate Cost function
# For every single data point, Python is doing a lot of work.

# def calculate_cost(x_train, y_train, b, w):
#     cost_total = 0
#     m = x_train.shape[0]
#     for i in range(m):
#         cost = ((x_train[i] * w + b) - y_train[i]) ** 2
#         cost_total = cost_total + cost
#     return (1/(2*m)) * cost_total


# Vectorized version of Calculate Cost function
# Vectorization is faster because it removes Python loops entirely
# and executes operations in compiled C code with CPU-level optimizations [SIMD (Single Instruction, Multiple Data) + caching).

def calculate_cost(x, y, b, w):
    m = len(x)
    prediction = w*x + b
    cost = np.sum((prediction-y)**2)
    return (1/(2*m)) * cost


# Loop version of Calculate Gradient function for univariate linear regression model
# For every single data point, Python is doing a lot of work.
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

# Vectorized version to Calculate Gradient for univariate linear regression model
def calculate_gradient(x, y, b, w):
    m = x.shape[0]
    prediction = w*x + b
    dj_dw = (1/m) * np.sum((prediction-y)*x)
    dj_db = (1/m) * np.sum(prediction-y)
    return dj_dw, dj_db


def gradient_descent(x, y, w_init, b_init, alpha, iterations):
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

        if i < 100000:
            J_history.append(calculate_cost(x, y, b, w))
            p_history.append([w, b])

        if i % math.ceil(iterations/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:5.4f}",
                  f"dj/dw: {dj_dw:0.2f} dj/db: {dj_db:0.2f}",
                  f"w : {w:0.2f}  b :{b:0.2f}")

    return w, b, J_history, p_history
