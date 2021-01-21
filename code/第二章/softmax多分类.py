import gzip

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
# print(train_lable.shape)
# plt.imshow(train_image[0])
# plt.show()

train_image = train_image / 255
test_image = test_image / 255

# 将顺序编码转为独热编码
train_label_onehot = tf.keras.utils.to_categorical(train_lable)
test_label_onehot = tf.keras.utils.to_categorical(test_label)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))  # 28*28
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss='categorical_crossentropy',
              metrics=['acc'],
              )
model.fit(train_image, train_label_onehot, epochs=5,validation_data=(test_image,test_label_onehot))
predict = model.predict(test_image)
