import gzip

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_data():
    # 文件获取
    train_image = r"../../dataset/fashion-mnist/train-images-idx3-ubyte.gz"
    test_image = r"../../dataset/fashion-mnist/t10k-images-idx3-ubyte.gz"
    train_label = r"../../dataset/fashion-mnist/train-labels-idx1-ubyte.gz"
    test_label = r"../../dataset/fashion-mnist/t10k-labels-idx1-ubyte.gz"  # 文件路径
    paths = [train_label, train_image, test_label, test_image]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)


# 加载数据
(train_image, train_lable), (test_image, test_label) = get_data()
# print(train_image.shape)
# 扩张维度
train_image = np.expand_dims(train_image, -1)
test_image = np.expand_dims(test_image, -1)
# print(train_image.shape)
model = tf.keras.Sequential()
# 添加卷积层
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=train_image.shape[1:], activation='relu', padding='same'))
# 添加池化层
model.add(tf.keras.layers.MaxPooling2D())
# 添加卷积层
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
# 添加全局平均池化层
model.add(tf.keras.layers.GlobalAveragePooling2D())
# 添加全连接层
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
history = model.fit(train_image, train_lable, epochs=30, validation_data=(test_image, test_label))
plt.plot(history.epoch, history.history.get('acc'), label='acc')
plt.plot(history.epoch, history.history.get('val_acc'), label='val_acc')
