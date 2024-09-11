import numpy as np
import col2im

cols = np.ones((4, 9))
img_shape = (1, 1, 4, 4)
images = col2im.col2im(cols, img_shape, 2, 2, 3, 3, 1, 0)
print(images)