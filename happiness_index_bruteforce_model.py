import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cost_function as cf
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Univariate Linear Regression\\gdp-vs-happiness.csv")

x = df['GDP per capita'].to_numpy()
y = df['Life satisfaction'].astype(float).to_numpy()
m = len(x)

# -----------------------------------
# Create grid of w and b values
# -----------------------------------
w_values = np.linspace(-0.000035, 0.000035, 1000)
b_values = np.linspace(4.65, 4.75, 1000)

W, B = np.meshgrid(w_values, b_values)

# Cost matrix
J = np.zeros_like(W)

# Compute cost for each (w, b)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        J[i, j] = cf.calculate_cost(x, y, B[i, j], W[i, j])


# Find index of minimum cost
min_index = np.unravel_index(np.argmin(J), J.shape)

# Get corresponding values
min_cost = J[min_index]
best_w = W[min_index]
best_b = B[min_index]

print(f"Minimum Cost: {min_cost:.2f}")
print(f"Best w: {best_w:.10f}")
print(f"Best b: {best_b:.10f}")

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

surface = ax1.plot_surface(
    W, B, J,
    cmap='viridis',
    edgecolor='none',
    alpha=0.8
)

ax1.set_xlabel('w')
ax1.set_ylabel('b')
ax1.set_zlabel('Cost J(w,b)')
ax1.set_title('Cost Function Surface')

fig.colorbar(surface, ax=ax1, shrink=0.5)

# -----------------------------------
# Contour Plot
# -----------------------------------
ax2 = fig.add_subplot(1, 2, 2)

contour = ax2.contour(
    W, B, J,
    levels=20,
    cmap='viridis'
)

# Mark the minimum cost point
ax2.plot(best_w, best_b, 'r*', markersize=15, label=f'Minimum (w={best_w:.10f}, b={best_b:.2f})')

ax2.set_xlabel('w')
ax2.set_ylabel('b')
ax2.set_title('Contour Plot of Cost Function')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_aspect('auto')
ax2.legend()

plt.tight_layout()
plt.show()
