import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# 通过pands读取Income1数据集
data = pd.read_csv('./dataset/Income1.csv')
print(data)

# 以Education为x轴，以Income为y轴绘制图
plt.scatter(data.Education, data.Income)
plt.show()

x = data.Education
y = data.Income

# 创建模型结构
model = tf.keras.Sequential()
# 添加模型
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
# 查看各层的参数情况
model.summary()  # ax + b
# 优化方法adam，损失函数均方差mse
model.compile(optimimsezer='adam',
              loss='')

# 使用fit方法训练，轮数5000
history = model.fit(x, y, epochs=5000)

# 预测
plt.scatter(x, model.predict(x))
plt.show()
