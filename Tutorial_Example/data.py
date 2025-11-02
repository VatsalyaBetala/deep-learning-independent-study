import numpy as np

def make_sine_data(N=100, sigma=0.1, seed=42):
    """Return x (N,), t_true=sin(2πx), t=t_true+noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, N)
    t_true = np.sin(2 * np.pi * x)
    t = t_true + rng.normal(0.0, sigma, size=x.shape)
    return x, t_true, t