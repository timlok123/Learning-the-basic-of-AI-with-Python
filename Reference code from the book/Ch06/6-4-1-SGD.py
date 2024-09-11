import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# -- 載入 Iris 資料集 --
iris_data = datasets.load_iris()
input_data = iris_data.data
correct = iris_data.target
n_data = len(correct)  # 樣本數

# -- 將訓練樣本做正規化 (Normalization) 處理 --
ave_input = np.average(input_data, axis=0)
std_input = np.std(input_data, axis=0)
input_data = (input_data - ave_input) / std_input

# -- 將標籤 (正確答案) 做 one-hot 編碼 --
correct_data = np.zeros((n_data, 3))
for i in range(n_data):
    correct_data[i, correct[i]] = 1.0

# -- 將資料集拆分為「訓練資料集」與「測試資料集」 --
index = np.arange(n_data)
index_train = index[index%2 == 0]
index_test = index[index%2 != 0]

input_train = input_data[index_train, :]  # 輸入
correct_train = correct_data[index_train, :]  # 正確答案
input_test = input_data[index_test, :]  # 輸入
correct_test = correct_data[index_test, :]  #  正確答案

n_train = input_train.shape[0]  # 訓練資料的樣本數 (75)
n_test = input_test.shape[0]  # 測試資料的樣本數 (75)



# -- 各設定値 --
n_in = 4  # 輸入層的神經元數量
n_mid = 25  # 中間層的神經元數量
n_out = 3  # 輸出層的神經元數量

wb_width = 0.1  # 設定權重參數的初始值乘上 0.01
eta = 0.01  # 學習率 (eta)
epoch = 1000
batch_size = 8
interval = 100


# -- 父類別 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)  # 權重矩陣
        self.b = wb_width * np.random.randn(n)  # 偏值向量
        
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) # ReLU
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLU的微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 

# -- 輸出層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # Softmax

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
        

# -- 各層的實體化 --
middle_layer_1 = MiddleLayer(n_in, n_mid)
middle_layer_2 = MiddleLayer(n_mid, n_mid)
output_layer = OutputLayer(n_mid, n_out)



# -- 前向傳播 --
def forward_propagation(x):
    middle_layer_1.forward(x)
    middle_layer_2.forward(middle_layer_1.y)
    output_layer.forward(middle_layer_2.y)

# -- 反向傳播 --
def backpropagation(t):
    output_layer.backward(t)
    middle_layer_2.backward(output_layer.grad_x)
    middle_layer_1.backward(middle_layer_2.grad_x)

# -- 修正權重參數 --
def update_wb():
    middle_layer_1.update(eta)
    middle_layer_2.update(eta)
    output_layer.update(eta)

# -- 計算誤差 --
def get_error(t, batch_size):
    return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size  # 交叉熵誤差


# -- 開始訓練 --

# -- 記錄誤差用 --
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

# -- 記錄學習與進度--
n_batch = n_train // batch_size  # 每 1 epoch 的批次數量

for i in range(epoch):

    
    # -- 計算誤差 --  
    forward_propagation(input_train)
    error_train = get_error(correct_train, n_train)
    forward_propagation(input_test)
    error_test = get_error(correct_test, n_test)
    
    # -- 記錄誤差 -- 
    test_error_x.append(i)
    test_error_y.append(error_test) 
    train_error_x.append(i)
    train_error_y.append(error_train) 
    
    # -- 顯示進度 -- 
    if i%interval == 0:
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))

       
    
    # -- 訓練 -- 
    index_random = np.arange(n_train)
    np.random.shuffle(index_random)  # 索引洗牌
    for j in range(n_batch):
        
        # 取出小批次
        mb_index = index_random[j*batch_size : (j+1)*batch_size]
        x = input_train[mb_index, :]
        t = correct_train[mb_index, :]
        
        # 前向傳播與反向傳播
        forward_propagation(x)
        backpropagation(t)
        
        # 更新權重與偏值
        update_wb()



        
# -- 以圖表顯示誤差記錄 -- 
plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()


# -- 計算準確率 -- 
forward_propagation(input_train)
count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))

forward_propagation(input_test)
count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_test, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%",
      "Accuracy Test:", str(count_test/n_test*100) + "%")