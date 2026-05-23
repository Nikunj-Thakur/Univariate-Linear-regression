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
    m = len(x))
    prediction = w*x + b
    cost = np.sum((prediction-y)**2)
    return (1/(2*m)) * cost
