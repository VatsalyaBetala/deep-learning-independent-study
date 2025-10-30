import numpy as np
import matplotlib.pyplot as plt

p = 0.7

x = [0, 1]
pmf = [1 - p, p]

plt.bar(x, pmf, color='royalblue', edgecolor='black', width=0.4)
plt.xticks(x, ['0 (Failure)', '1 (Success)'])
plt.ylabel('Probability')
plt.title(f'Bernoulli Distribution (p = {p})')
plt.ylim(0, 1)
plt.show()
