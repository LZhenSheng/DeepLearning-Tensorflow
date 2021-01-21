from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import glob
def load_img(path):
    # 通过tensorflow的read_file读取二进制文件
    img_raw = tf.io.read_file(all_image_path[3])
    # 对二进制图片进行解码
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3)
    #将所有图片转换成相同的大小
    img_tensor=tf.image.resize(img_tensor,[256,256])
    # 查看图片的形状
    # print(img_tensor.shape)
    # 查看图片的数据类型
    # print(img_tensor.dtype)
    # 将uint类型转换成float32类型
    img_tensor = tf.cast(img_tensor, tf.float32)
    # 将图片归一化
    img_tensor = img_tensor / 255
    # print(img_tensor.numpy().max())
    return img_tensor
#通过glob获取所有图片的路径
all_image_path=glob.glob('..\\..\\dataset\\2_class\\*\\*.jpg')
#多数据进行乱序
random.shuffle(all_image_path)

label_to_index={'airplane':0,'lake':1}
index_to_label=dict((v,k) for k,v in label_to_index.items())

#获取所有图片的标签
all_labels=[label_to_index.get(img.split('\\')[4]) for img in all_image_path]
#随机选取一个数
# i=random.choice(range(len(all_image_path)))
# img=all_image_path[i]
# label=all_labels[i]
# img_tensor=load_img(img)
#查看图片
# plt.title(index_to_label.get(label))
# plt.imshow(img_tensor.numpy())
# plt.show()
img_ds=tf.data.Dataset.from_tensor_slices(all_image_path)
label_ds=tf.data.Dataset.from_tensor_slices(all_labels)
#通过map方法处理所有的图片
img_ds=img_ds.map(load_img)
#和数据合并
img_label_ds=tf.data.Dataset.zip((img_ds,label_ds))
#划分训练数据和测试数据
img_count=len(all_image_path)
test_count=int(img_count*0.2)
train_count=img_count-test_count
train_ds=img_label_ds.skip(test_count)
test_ds=img_label_ds.take(test_count)
BATCH_SIZE=16
train_ds=train_ds.repeat().shuffle(100).batch(BATCH_SIZE)
test_ds=test_ds.batch(BATCH_SIZE)

#模型创建
model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(256,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(1024,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
print(model.summary())
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['acc'])
steps_per_epoch=train_count//BATCH_SIZE
val_step=test_count//BATCH_SIZE
history=model.fit(train_ds,epochs=10,steps_per_epoch=steps_per_epoch,validation_data=test_ds,validation_steps=val_step)
_test=all_image_path[10]
_test=load_img(_test)
_test=tf.expand_dims(_test,0)
if model.predict(_test)[0][0]>0.5:
    print(index_to_label[1])
else:
    print(index_to_label[0])