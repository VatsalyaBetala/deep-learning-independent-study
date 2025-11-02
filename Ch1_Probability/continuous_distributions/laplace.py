import numpy as np
import matplotlib.pyplot as plt

# --- Laplace with μ = 1, γ = 1 ---
x = np.linspace(-3, 4, 400)
mu, gamma = 1, 1
p_laplace = (1 / (2 * gamma)) * np.exp(-np.abs(x - mu) / gamma)

plt.figure(figsize=(6,4))
plt.plot(x, p_laplace, color='green')
plt.title('Laplace distribution (μ = 1, γ = 1)')
plt.xlabel('x'); plt.ylabel('p(x)')
plt.ylim(0, 1.2)
plt.grid(True)
plt.show()