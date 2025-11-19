import numpy as np

def design_matrix(x, M):
    return np.column_stack([x**j for j in range(M + 1)])