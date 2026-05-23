import matplotlib.pyplot as plt
import numpy as np

x = np.array([0, 1])
y = np.array([0, 1])

u = np.array([1, 0])   # horizontal movement
v = np.array([0, 1])   # vertical movement

# mag = np.sqrt(u**2 + v**2)

plt.quiver(x, y, u, v, 
           angles='xy',
           scale_units='xy',
           scale=1)

plt.xlim(-0.2, 5)
plt.ylim(-0.2, 5)

plt.grid()
plt.show()