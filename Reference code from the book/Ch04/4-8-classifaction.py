import numpy as np

x_0 = np.arange(-1.0, 1.0, 0.1)
x_1 = np.arange(-1.0, 1.0, 0.1)

Z=np.zeros((400,2))

w_im = np.array([[1.0,2.0],
                 [2.0,3.0]])
w_mo = np.array([[-1.0,1.0],
                 [1.0,-1.0]])

b_im = np.array([0.3,-0.3])
b_mo = np.array([0.4, 0.1])

def middle_layer(x, w, b):
    u = np.dot(x, w) + b
    return 1/(1+np.exp(-u))

def output_layer(x, w, b):
    u = np.dot(x, w) + b
    return np.exp(u)/np.sum(np.exp(u))

for i in range(20):
    for j in range(20):
        inp = np.array([x_0[i], x_1[j]])
        mid = middle_layer(inp, w_im, b_im)
        out = output_layer(mid, w_mo, b_mo)
        Z[i*20 + j] = out        
print(Z)

        
#透過 Z 的結果將 400 組 xy 組合分類，並畫成圖

plus_x = []
plus_y = []
circle_x = []
circle_y = []

for i in range(20):
    for j in range(20):
        if Z[i*20 +j][0] > Z[i*20 +j][1]:
            plus_x.append(x_0[i])
            plus_y.append(x_1[j])
        else:
            circle_x.append(x_0[i])
            circle_y.append(x_1[j])

import matplotlib.pyplot as plt
plt.scatter(plus_x, plus_y, marker="+")
plt.scatter(circle_x, circle_y, marker="o")
plt.show()
