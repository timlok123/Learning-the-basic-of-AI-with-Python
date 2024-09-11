import numpy as np 
import im2col 

img = np.arange(54).reshape(2,3,3,3)

cols = im2col.im2col(img, 2, 2, 2, 2, 1, 0)
print(cols)