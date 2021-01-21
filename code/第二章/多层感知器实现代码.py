import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取数据集
data = pd.read_csv("../../dataset/Advertising.csv")

# 读取前五行数据
# print(data.head())

# 特征值散点图
# plt.scatter(data.TV, data.sales)
# 图显示
# plt.show()

# 取所有行的第二列到倒数第二列所有数据
x = data.iloc[:, 1:-1]

# 取所有行的最后一列数据
y = data.iloc[:, -1]

# 建模模型
model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
                             tf.keras.layers.Dense(1)])

# 查看模型
# print(model.summary())

# 优化方法adam，损失函数均方差mse
model.compile(optimizer='adam', loss='mse')

# 使用fit方法训练，轮数5000
model.fit(x, y, epochs=5000)

# 获取测试数据
test_x = data.iloc[:10, 1:-1]
test_y = data.iloc[:10, -1]

# 查看预测数据和实际数据区别
result = model.predict(test_x)
for i in range(10):
    print(test_y[i], "-", result[i][0])
plt.show()
