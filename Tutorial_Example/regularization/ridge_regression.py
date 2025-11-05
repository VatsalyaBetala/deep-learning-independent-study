import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

np.random.seed(42)
x = np.linspace(0, 1, 100)
t_true = np.sin(2 * np.pi * x)
t = t_true + np.random.normal(0, 0.1, x.shape)

def design_matrix(x, M):
    return np.vstack([x**i for i in range(M + 1)]).T

M = 9
Phi = design_matrix(x, M)

lam = 1e-2 
ridge = Ridge(alpha=lam, fit_intercept=False)
ridge.fit(Phi, t)
w = ridge.coef_

x_test = np.linspace(0, 1, 200)
Phi_test = design_matrix(x_test, M)
y_pred = Phi_test @ w
y_true = np.sin(2 * np.pi * x_test)

rmse = np.sqrt(np.mean((Phi @ w - t_true) ** 2))
print(f"Ridge Regression (λ={lam}) — RMSE (true): {rmse:.4f}")

plt.figure(figsize=(7,5))
plt.scatter(x, t, color='blue', s=16, label='Training data')
plt.plot(x_test, y_true, color='green', label='True function: sin(2πx)')
plt.plot(x_test, y_pred, color='red', label=f'Ridge fit (λ={lam})')
plt.legend(); plt.xlabel('x'); plt.ylabel('t')
plt.title(f'Ridge Regression (Polynomial degree M={M})')
plt.tight_layout(); plt.show()
