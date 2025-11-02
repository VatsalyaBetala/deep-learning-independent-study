import numpy as np

def design_matrix(x, M):
    """Φ[n, j] = x_n ** j for j=0..M. Returns shape (N, M+1)."""
    x = np.asarray(x).reshape(-1)
    return np.vstack([x**j for j in range(M+1)])