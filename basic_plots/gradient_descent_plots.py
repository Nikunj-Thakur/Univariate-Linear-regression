import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Example training data
# ---------------------------------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([300, 500, 700, 900, 1100])

m = len(x)

# ---------------------------------------------------
# Cost function
# ---------------------------------------------------
def compute_cost(w, b):
    predictions = w * x + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# ---------------------------------------------------
# Partial derivatives
# ---------------------------------------------------
def compute_gradient(w, b):

    predictions = w * x + b

    dj_dw = (1 / m) * np.sum((predictions - y) * x)
    dj_db = (1 / m) * np.sum(predictions - y)

    return dj_dw, dj_db

# ===================================================
# 1. COST vs w GRAPH
# ===================================================

b_fixed = 100

w_array = np.linspace(0, 400, 200)
cost_array = []

for w in w_array:
    cost_array.append(compute_cost(w, b_fixed))

plt.figure(figsize=(8,6))
plt.plot(w_array, cost_array, color='dodgerblue')

# We are taking just three Sample points from range 0 to 400 for plotting gradient at these points
sample_ws = [100, 200, 300]

for w in sample_ws:

    cost = compute_cost(w, b_fixed)
    dj_dw, _ = compute_gradient(w, b_fixed)

    plt.scatter(w, cost, color='r')

    plt.text(
        w + 5,
        cost,
        f"dJ/dw = {dj_dw:.0f}",
        fontsize=11
    )

plt.xlabel("w")
plt.ylabel("Cost")
plt.title("Cost vs w, with b fixed to 100")

plt.grid(True)
plt.show()

# ===================================================
# 2. QUIVER PLOT OF GRADIENTS
# ===================================================

w_vals = np.linspace(-100, 600, 12)
b_vals = np.linspace(-200, 200, 10)

W, B = np.meshgrid(w_vals, b_vals)

U = np.zeros_like(W)   # dJ/dw
V = np.zeros_like(B)   # dJ/db

# Compute gradients at each point
for i in range(W.shape[0]):
    for j in range(W.shape[1]):

        dj_dw, dj_db = compute_gradient(W[i, j], B[i, j])

        U[i, j] = dj_dw
        V[i, j] = dj_db

# Normalize arrows for better display
magnitude = np.sqrt(U**2 + V**2)

U = U / magnitude
V = V / magnitude

plt.figure(figsize=(10,6))

plt.quiver(W, B, U, V, magnitude)

plt.xlabel("w")
plt.ylabel("b")
plt.title("Gradient shown in quiver plot")

plt.grid(True)
plt.show()