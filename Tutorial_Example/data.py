import numpy as np

def make_sine_data(N=10, sigma=0.1):
    np.random.seed(0) 
    x = np.linspace(0, 1, N)
    t_true = np.sin(2 * np.pi * x)
    noise = np.random.normal(0, sigma, size=N)
    t = t_true + noise

    return x, t_true, t