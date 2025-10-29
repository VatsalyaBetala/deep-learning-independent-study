import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

N = 10
mu = 0.25

m = np.arange(0, N + 1)
p = binom.pmf(m, N, mu)

plt.bar(m, p, color='blue', edgecolor='black')
plt.title(f'Binomial Distribution (N={N}, μ={mu})')
plt.xlabel('m')
plt.ylabel('Probability')
plt.xticks(np.arange(0, N + 1))
plt.show()
