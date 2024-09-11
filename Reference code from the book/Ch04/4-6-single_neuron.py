
import numpy as np

x_0 = np.arange(-1.0, 1.0, 0.2)  
x_1 = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros(100) 

w_x_0 = 2.5
w_x_1 = 3.0

bias = 0.1

for i in range(10):
    for j in range(10):
        u = x_0[i] * w_x_0 + x_1[j] * w_x_1 + bias  
        y = 1/(1+np.exp(-u)) 
        Z[i *10 + j] = y
print(Z)

import matplotlib.pyplot as plt
plt.imshow(Z.reshape(10,10), "gray", vmin = 0.0, vmax =1.0)
plt.colorbar()  
plt.show()
