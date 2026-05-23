import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2*np.pi, 2*np.pi, 1000)
y = np.linspace(0, 9, 10)
print(y)

sin = np.sin(x)
cos = np.cos(x)
tan = np.tan(x)
cot = 1/tan
csc = 1/sin
sec = 1/cos

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

ax1.plot(x, sin)
ax1.set_title("Sine Function")
ax1.grid(True)

ax2.plot(x, cos)
ax2.set_title("Cosine Function")
ax2.grid(True)

ax3.plot(x, tan)
ax3.set_title("Tangent Function")
ax3.set_ylim(-10, 10)
ax3.grid(True)

ax4.plot(x, csc)
ax4.set_title("Cosecant Function")
ax4.set_ylim(-10, 10)
ax4.grid(True)

ax5.plot(x, sec)
ax5.set_title("Secant Function")
ax5.set_ylim(-10, 10)
ax5.grid(True)

ax6.plot(x, cot)
ax6.set_title("Cotangent Function")
ax6.set_ylim(-10, 10)
ax6.grid(True)

plt.tight_layout()
#plt.show()