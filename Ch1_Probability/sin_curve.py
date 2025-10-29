import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n = 100
x1 = np.random.uniform(0, 1, n)
x2 = np.random.uniform(0, 1, n)

def f(x1, x2):
    return np.sin(2 * np.pi * x1) * np.sin(2 * np.pi * x2)

noise = 0.1 * np.random.randn(n)
y = f(x1, x2) + noise

X1, X2 = np.meshgrid(np.linspace(0,1,50), np.linspace(0,1,50))
Y = f(X1, X2)

fig = plt.figure(figsize=(14,5))
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(X1, X2, Y, color='brown', alpha=0.8)
ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$y$')
ax.set_title('True function surface')

# 3. Plot (b) noisy projection (x2 unobserved)
ax2 = fig.add_subplot(132)
ax2.scatter(x1, y, color='red')
ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$y$')
ax2.set_title('x2 unobserved (high noise)')

# 4. Plot (c) fixed x2
x2_fixed = np.pi/2 / (2*np.pi)  # scale to [0,1] domain
x1_fixed = np.random.uniform(0, 1, n)
y_fixed = f(x1_fixed, np.full_like(x1_fixed, x2_fixed)) + noise

ax3 = fig.add_subplot(133)
ax3.scatter(x1_fixed, y_fixed, color='red')
ax3.set_xlabel('$x_1$'); ax3.set_ylabel('$y$')
ax3.set_title('x2 fixed = π/2 (low noise)')

plt.tight_layout()
plt.show()
