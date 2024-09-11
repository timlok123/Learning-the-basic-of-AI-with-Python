
import matplotlib.pyplot as plt 
from sklearn import datasets

digits = datasets.load_digits()
print(digits.data.shape)

plt.imshow(digits.data[80].reshape(8, 8), cmap="gray")
plt.show() 


#-------------------------


print(digits.target.shape)
print(digits.target[:50])


#-------------------------



import matplotlib.pyplot as plt 
from sklearn import datasets

digits = datasets.load_digits()

fig = plt.figure()
for i in range(10):
    ax = fig.add_subplot(2, 5, i+1)
    ax.tick_params(labelbottom="off",bottom="off")
    ax.tick_params(labelleft="off",left="off") 
    plt.imshow(digits.data[i].reshape(8, 8), cmap="gray")
    plt.title(digits.target[i])

plt.show()