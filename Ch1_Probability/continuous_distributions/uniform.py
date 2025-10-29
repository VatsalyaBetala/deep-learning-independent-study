import numpy as np
import matplotlib.pyplot as plt

c, d = 0, 5
x = np.linspace(-1,6,500)
p = np.where((x>=c)&(x<=d), 1/(d-c), 0)
plt.plot(x,p,'r',lw=2)
plt.fill_between(x,p,0,alpha=0.2)
plt.xlabel('x'); plt.ylabel('p(x)')
plt.title('Uniform Distribution')
plt.show()
