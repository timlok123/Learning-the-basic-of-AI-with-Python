import numpy as np
import matplotlib.pyplot as plt

x_0 = np.arange(-1.0, 1.0, 0.2)
x_1 = np.arange(-1.0, 1.0, 0.2)

Z = np.zeros(100)

w_im = np.array([[4.0,4.0],
                 [4.0,4.0]])
w_mo = np.array([[1.0],
                 [-1.0]])

b_im = np.array([3.0,-3.0])
b_mo = np.array([0.1])

def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))

def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return u  

for i in range(10):
    for j in range(10):
        inp = np.array([x_0[i], x_1[j]])       
        mid = middle_layer(inp, w_im, b_im) 
        out = output_layer(mid, w_mo, b_mo)
        Z[i*10+j] = out[0]


plt.imshow(Z.reshape(10,10), "gray", vmin = 0.0, vmax = 1.0)
plt.colorbar()  
plt.show()