import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# 没有头部数据,header=None
data = pd.read_csv('../../dataset/credit-a.csv', header=None)

# 查看头部数据
# print(data.head())

# 查看目标列值
# print(data.iloc[:, -1].value_counts())

x = data.iloc[:, :-1]

# 获取目标数据并用0替换-1
y = data.iloc[:, -1].replace(-1, 0)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x, y, epochs=5000)

plt.plot(history.epoch, history.history.get('loss'))
plt.show()
plt.plot(history.epoch, history.history.get('acc'))
plt.show()
