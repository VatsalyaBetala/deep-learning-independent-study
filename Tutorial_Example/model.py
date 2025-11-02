import numpy as np
import matplotlib.pyplot as plt
from basis import design_matrix
from data import make_sine_data

x, t_true, t = make_sine_data(N=10, sigma=0.1)

# Test data
x_test = np.linspace(0, 1, 100)
y_test = np.sin(2 * np.pi * x_test)

# Define degrees to test
degrees = [0, 1, 2, 3, 4, 9, 15]

for i in degrees:
    Phi_train = design_matrix(x, i) # (N, M+1)
    Phi_test = design_matrix(x_test, i) # (100, M+1)
    
    w_star = np.linalg.inv(Phi_train.T @ Phi_train) @ Phi_train.T @ t # (M+1, 1) 
    
    # (100, M+1) @ (M+1, 1)
    y_pred = Phi_test @ w_star # (100, 1)

    plt.figure(figsize=(6,4))
    plt.scatter(x, t, color='blue', label='Training data')
    plt.plot(x_test, y_test, color='green', label='True function: sin(2πx)')
    plt.plot(x_test, y_pred, color='red', label=f'Fitted polynomial (M={i})')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Polynomial Regression Fit (Degree M={i})')
    plt.show()

    # model predictions on the training inputs
    y_train_pred = Phi_train @ w_star

    # True values for same inputs
    t_true_train = np.sin(2 * np.pi * x)

    # RMSE 
    rmse_true = np.sqrt(np.mean((y_train_pred - t_true_train)**2))
    print(f"RMSE vs True Function (M={i}): {rmse_true:.5f}")
