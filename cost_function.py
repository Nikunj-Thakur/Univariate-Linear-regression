def calculate_cost(x_train, y_train, b, w):
    cost_total = 0
    m = x_train.shape[0]
    for i in range(m):
        cost = ((x_train[i] * w + b) - y_train[i]) ** 2
        cost_total = cost_total + cost
    return (1/(2*m)) * cost_total
