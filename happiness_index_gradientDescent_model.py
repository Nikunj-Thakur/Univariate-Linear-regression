import numpy as np
import utility_functions as uf
import pandas as pd
import matplotlib.pyplot as plt

# # Load our data set
# x_train = np.array([1.0, 2.0])   #features
# y_train = np.array([300.0, 500.0])   #target value

import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Univariate Linear Regression\\WHR_2024.csv")
df = df.dropna() 
x_train = df['gdp_per_capita'].to_numpy()
y_train = df['happiness_score'].astype(float).to_numpy()

# Scaling the feture using Z-score standardisation to prevent overflow
x_mean = x_train.mean()
x_std = x_train.std()
x_train = (x_train - x_mean) / x_std


# initialize parameters randomly with initial values as zeros
w_init = 0
b_init = 0

# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2

# run gradient descent
w_final, b_final, J_hist, p_hist = uf.gradient_descent(
    x_train, y_train, w_init, b_init, tmp_alpha, iterations)
print(f"(w,b) found by gradient descent: ({w_final:0.3f},{b_final:0.3f})")

# plot cost versus iteration for first 100 steps
fig = plt.figure(figsize=(12, 5))
ax1=fig.add_subplot(131)

ax1.plot(J_hist[:100]) #ax1.plot(np.arange(len(J_hist[:100])), J_hist[:100])
ax1.set_title("Cost vs. iteration(Start-First 100 Iteration)");  
ax1.set_ylabel('Cost')  
ax1.set_xlabel('iteration step')

# plot cost versus iteration from 1,000 steps to 10,000 steps
ax2=fig.add_subplot(132)
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax2.set_title("Cost vs. iteration (End - From 1000 to 10000 Iterations)")
ax2.set_ylabel('Cost') 
ax2.set_xlabel('iteration step') 
ax2.ticklabel_format(style='plain', axis='y', useOffset=False)


# plot Gradient descent path on the contour plot
ax3=fig.add_subplot(133)

w_history = [p[0] for p in p_hist]
b_history = [p[1] for p in p_hist]

# Create grid of w and b values
w_vals = np.linspace(min(w_history) - 1, max(w_history) + 1, 100)
b_vals = np.linspace(min(b_history) - 1, max(b_history) + 1, 100)

W, B = np.meshgrid(w_vals, b_vals)

# Compute cost for each (w, b)
Z = np.zeros_like(W)

m = len(x_train)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        w = W[i, j]
        b = B[i, j]
        y_pred = w * x_train + b
        Z[i, j] = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)

# Contour plot
contour=ax3.contour(W, B, Z, levels=40,cmap='viridis')
ax3.plot(w_history, b_history, 'r.-', markersize=6, linewidth=1.5)
ax3.set_xlabel("w")
ax3.set_ylabel("b")
ax3.set_title("Cost contour with Gradient Descent path")
ax3.clabel(contour, inline=True, fontsize=8)
ax3.set_aspect('auto')



plt.tight_layout()
plt.show()

