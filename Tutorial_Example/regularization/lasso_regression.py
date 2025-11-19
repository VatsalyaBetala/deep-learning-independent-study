import numpy as np
import matplotlib.pyplot as plt

def design_matrix(x, M):
    return np.vstack([x**i for i in range(M + 1)]).T

def run_ridge_poly(N=10, sigma=0.1, M=9, lam=0, seed=42):

    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, N)
    t_true = np.sin(2 * np.pi * x)
    t = t_true + rng.normal(0, sigma, size=x.shape)

    Phi = design_matrix(x, M)

    I = np.eye(M+1)
    A = Phi.T @ Phi + lam * I
    b = Phi.T @ t
    w = np.linalg.solve(A, b)

    x_test = np.linspace(0, 1, 400)
    Phi_test = design_matrix(x_test, M)
    y_pred = Phi_test @ w
    y_true = np.sin(2 * np.pi * x_test)

    plt.figure(figsize=(7,5))
    plt.scatter(x, t, color='blue', s=40, label='Training data')
    plt.plot(x_test, y_true, color='green', linewidth=2, label='True function')
    plt.plot(x_test, y_pred, color='red', linewidth=2,
             label=f'Ridge fit (ln λ = {np.log(lam) if lam>0 else "-∞"})')

    plt.xlabel("x")
    plt.ylabel("t")
    plt.legend()
    plt.title(f"Ridge Regression (M={M}, λ={lam})")
    plt.tight_layout()
    plt.show()

# λ -> 0 (no regularization, ln λ → –∞)
run_ridge_poly(N=10, sigma=0.1, M=9, lam=0)

# λ = exp(-18)
run_ridge_poly(N=10, sigma=0.1, M=9, lam=np.exp(-18))

# λ = 1  (ln λ = 0)
run_ridge_poly(N=10, sigma=0.1, M=9, lam=1)