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

train_image = train_image / 255
test_image = test_image / 255

ds_train_img = tf.data.Dataset.from_tensor_slices(train_image)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_lable)

# 通过zip函数将ds_train_img和ds_train_lab合并到一起
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lab))
ds_test = tf.data.Dataset.from_tensor_slices((test_image, test_label))
ds_train = ds_train.shuffle(10000).repeat().batch(64)
ds_test = ds_test.batch(64)
model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                             tf.keras.layers.Dense(128, activation='relu'),
                             tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])
# train_image第一个维度时总的图片数量大小
steps_per_epoch = train_image.shape[0] // 64
valication_epoch = test_image.shape[0] // 64
model.fit(ds_train, epochs=5, steps_per_epoch=steps_per_epoch, validation_data=ds_test,
          validation_steps=valication_epoch)
