import numpy as np
import matplotlib.pyplot as plt
from basis import design_matrix
from data import make_sine_data

def run_polynomial_regression(N=10, sigma=0.1):

    x, t_true, t = make_sine_data(N=N, sigma=sigma)

    x_test = np.linspace(0, 1, 200)
    y_test = np.sin(2 * np.pi * x_test)

    degrees = [0, 1, 3, 9]

    for M in degrees:

        Phi_train = design_matrix(x, M)
        Phi_test = design_matrix(x_test, M)

        w_star = np.linalg.solve(Phi_train.T @ Phi_train, Phi_train.T @ t)

        y_pred_test = Phi_test @ w_star

        plt.figure(figsize=(6, 4))
        plt.scatter(x, t, color='blue', label='Training data')
        plt.plot(x_test, y_test, color='green', label='True sin(2πx)')
        plt.plot(x_test, y_pred_test, color='red', label=f'Poly fit (M={M})')

        plt.xlabel("x")
        plt.ylabel("t")
        plt.title(f"Polynomial Regression (M={M})")
        plt.legend()
        plt.show()

def run_error_curves(N_train=10, N_test=100, sigma=0.1):

    x_train, t_true_train, t_train = make_sine_data(
        N=N_train, sigma=sigma
    )

    x_test = np.linspace(0, 1, N_test)
    t_test = np.sin(2 * np.pi * x_test)

    degrees = list(range(10))  # M = 0 ... 9
    train_rmse = []
    test_rmse = []

    for M in degrees:

        Phi_train = design_matrix(x_train, M)
        Phi_test = design_matrix(x_test, M)

        A = Phi_train.T @ Phi_train
        b = Phi_train.T @ t_train
        w_star = np.linalg.solve(A, b)

        y_train_pred = Phi_train @ w_star
        y_test_pred = Phi_test @ w_star

        train_rmse.append(
            np.sqrt(np.mean((y_train_pred - t_train)**2))
        )

        test_rmse.append(
            np.sqrt(np.mean((y_test_pred - t_test)**2))
        )

    plt.figure(figsize=(7, 4))
    plt.plot(degrees, train_rmse, 'o-', color='red', label='Training')
    plt.plot(degrees, test_rmse, 'o-', color='blue', label='Test')

    plt.xlabel("M (polynomial degree)")
    plt.ylabel(r"$E_{RMS}$")
    plt.title("Training vs Test RMS Error (Bishop Figure 1.7)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()

run_polynomial_regression(N=10, sigma=0.1)
run_error_curves()
