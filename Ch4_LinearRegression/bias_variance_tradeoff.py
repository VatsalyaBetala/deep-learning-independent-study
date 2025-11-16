import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Generate true function
def f(x): 
    return np.sin(2 * np.pi * x)

# Gaussian basis functions
def design_matrix(x, mus, s):
    Phi = np.exp(-0.5 * ((x[:, None] - mus[None, :]) / s) ** 2)
    Phi = np.concatenate([Phi, np.ones((len(x), 1))], axis=1)  # add bias
    return Phi

N = 25                  # points per dataset
L = 100                 # number of datasets
M = 24                  # number of Gaussian bases
mus = np.linspace(0, 1, M)
s = 0.1
x_plot = np.linspace(0, 1, 200)
Phi_plot = design_matrix(x_plot, mus, s)

# --- Regularization values ---
ln_lams = [3, 1, -3]
lams = [np.exp(l) for l in ln_lams]

# --- Create figure ---
fig, axes = plt.subplots(len(lams), 2, figsize=(8, 8))
for i, lam in enumerate(lams):
    fits = []
    for _ in range(L):
        x = np.random.rand(N)
        t = f(x) + np.random.normal(0, 0.1, N)
        Phi = design_matrix(x, mus, s)
        I = np.eye(Phi.shape[1])
        w = np.linalg.solve(Phi.T @ Phi + lam * I, Phi.T @ t)
        y = Phi_plot @ w
        fits.append(y)
    fits = np.array(fits)
    
    for y in fits[:20]:
        axes[i, 0].plot(x_plot, y, "r-", lw=0.6)
    axes[i, 0].set_ylim(-1.5, 1.5)
    axes[i, 0].set_xlim(0, 1)
    axes[i, 0].text(0.05, 1.2, f"ln λ = {ln_lams[i]}")
    axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

    mean_fit = fits.mean(axis=0)
    axes[i, 1].plot(x_plot, f(x_plot), "g", lw=1.5)
    axes[i, 1].plot(x_plot, mean_fit, "r", lw=1.5)
    axes[i, 1].set_ylim(-1.5, 1.5)
    axes[i, 1].set_xlim(0, 1)
    axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

plt.tight_layout()
plt.show()
