import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

def design_matrix(x, M):
    """Create a polynomial design matrix up to degree M."""
    return np.vstack([x**i for i in range(M + 1)]).T

def run_lasso_polynomial(N=100, sigma=0.1, M=9, alpha=1e-3, random_seed=42):
    np.random.seed(random_seed)

    x = np.linspace(0, 1, N)
    t_true = np.sin(2 * np.pi * x)
    t = t_true + np.random.normal(0, sigma, x.shape)

    Phi = design_matrix(x, M)

    scaler = StandardScaler()
    Phi_scaled = scaler.fit_transform(Phi)

    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    lasso.fit(Phi_scaled, t)
    w = lasso.coef_

    x_test = np.linspace(0, 1, 200)
    Phi_test = design_matrix(x_test, M)
    Phi_test_scaled = scaler.transform(Phi_test)

    y_pred = Phi_test_scaled @ w
    y_true = np.sin(2 * np.pi * x_test)

    rmse = np.sqrt(np.mean((Phi_scaled @ w - t_true)**2))
    print(f"Lasso Regression (N={N}, σ={sigma}, M={M}, α={alpha}) — RMSE: {rmse:.4f}")

    plt.figure(figsize=(7, 5))
    plt.scatter(x, t, color='blue', s=40, label='Training data')
    plt.plot(x_test, y_true, color='green', label='True function: sin(2πx)')
    plt.plot(x_test, y_pred, color='orange', label=f'Lasso fit (α={alpha})')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.title(f'Lasso Regression (N={N}, σ={sigma}, M={M})')
    plt.tight_layout()
    plt.show()

run_lasso_polynomial(N=10, sigma=0.1, M=9, alpha=1e-2)

# run_lasso_polynomial(N=30, sigma=0.1, M=9, alpha=1e-3)

# run_lasso_polynomial(N=100, sigma=0.1, M=9, alpha=1e-4)

# run_lasso_polynomial(N=30, sigma=0.3, M=9, alpha=1e-2)

# run_lasso_polynomial(N=30, sigma=0.1, M=9, alpha=1e-3)