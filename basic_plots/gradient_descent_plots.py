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

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1)

ax1.plot(w_array, cost_array, color='b')

# We are taking just three Sample points from range 0 to 400 for plotting gradient at these points
sample_ws = [100, 200, 300]

for w in sample_ws:

    cost = compute_cost(w, b_fixed)
    dj_dw, _ = compute_gradient(w, b_fixed)

    ax1.scatter(w, cost, color='r')

    ax1.text(
        w + 5,
        cost,
        f"dJ/dw = {dj_dw:.0f}",
        fontsize=11
    )

ax1.set_xlabel("w")
ax1.set_ylabel("Cost")
ax1.set_title("Cost vs w, with b fixed to 100")
ax1.grid(True)


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


magnitude = np.sqrt(U**2 + V**2)

# We can Normalize arrows using Unit Vector Scaling for better display, but here I want to display true magnitudes of gradients,
# so I am not using it
# U = U / magnitude
# V = V / magnitude

ax2 = fig.add_subplot(1, 2, 2)
ax2.quiver(W, B, U, V, magnitude)

ax2.set_xlabel("w")
ax2.set_ylabel("b")
ax2.set_title("Gradient shown in quiver plot")
ax2.grid(True)

# Leave space at the bottom
plt.subplots_adjust(bottom=0.25)


fig.text(
    0.1,                # x position
    0.02,               # y position
    "Above, the left plot shows  ∂J(w,b)/∂w or the slope of the cost curve relative to w at three points. "
    "On the right side of the plot, the derivative is positive, while on the left it is negative."
    " Due to the 'bowl shape', the derivatives will always lead gradient descent toward the bottom where the gradient is zero."
    "The left plot has fixed  b=100. Gradient descent will utilize both  ∂J(w,b)/∂w and  ∂J(w,b)/∂b to update parameters."

    "\n\nThe 'quiver plot' on the right provides a means of viewing the gradient of both parameters. The arrow sizes reflect the magnitude of the gradient at that point. "
    "The direction and slope of the arrow reflects the ratio of  ∂J(w,b)/∂b and  ∂J(w,b)/∂w at that point. "
    "Note that the gradient points away from the minimum. "
    "The scaled gradient is subtracted from the current value of w or b. This moves the parameter in a direction that will reduce cost.",

    ha='left',
    fontsize=10,
    wrap=True,
    color='darkblue',
    bbox=dict(
        facecolor='lightyellow',
        edgecolor='black',
        boxstyle='round,pad=0.5',
        linewidth=2
    )
)

plt.show()
