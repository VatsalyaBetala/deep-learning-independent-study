import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,500)
for l in [0.5,1,2]:
    plt.plot(x, l*np.exp(-l*x), label=f'λ={l}')
plt.xlabel('x'); plt.ylabel('p(x|λ)')
plt.legend(); plt.title('Exponential Distribution')
plt.show()
