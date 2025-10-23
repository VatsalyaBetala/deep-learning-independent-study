import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 400)
p = norm.pdf(x, 0, 1)
P = norm.cdf(x, 0, 1)

plt.figure(figsize=(8,4))
plt.plot(x, p, 'r', label='p(x) PDF')
plt.plot(x, P, 'b', label='P(x) CDF')
plt.fill_between(x, p, 0, where=(x>-0.5)&(x<0.5), color='lime', alpha=0.3)
plt.xlabel('x'); plt.legend(); plt.title('Density vs Cumulative')
plt.show()
