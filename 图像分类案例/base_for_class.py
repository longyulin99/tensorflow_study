#此函数用来训练第一个神经网络：基本分类模型
#实现训练神经网络模型，对运动鞋和衬衫等服装图像做基本分类
#本神经网络使用tf.keras
import tensorflow as tf
from tensorflow import keras
import pandas as pd

#辅助模型
import numpy as np
import matplotlib.pyplot as plt

#首先导入Fashion MNIST数据集（10个类别70000个灰度图像）
#分成60000个训练及图像和10000个测试集图像
#直接导入数据（注意输入空格）
fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#显示train_images与train_labels
# print(train_images)
# print(train_labels)


#创建映射关系
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#对数据进行预处理
plt.figure()     #开启一个窗口
plt.imshow(train_images[0])     #显示图片
plt.colorbar()          #将颜色条加入到绘图中
plt.gca().grid(False)       #获取当前图标，然后设置不显示背景的网络线


#将数据的值缩放到0~1
train_images = train_images/255.0
train_labels = train_labels/255.0

#建立模型
#图层提供图像的数据表示
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),    #只是将图像的格式从2d阵列转化为1d的784像素的向量表示,没有需要学习的参数，只是重新格式化数据
    keras.layers.Dense(128, activation=tf.nn.relu),         #这两层为全连接神经层，第一层有128个节点(说明第一层的w有128。每一个都是784的向量)，第二层是10个softmax（）节点，返回10个概率分布的数组
    keras.layers.Dense(10, activation=tf.nn.softmax)    #一个神经元是由多个W组成，w的个数决定了下一个输出的值的个数，如一个层有128个w，则输出后有128个输出值
])




#编译模型：
#需要添加损失函数，希望最小化这个值
#优化器：基于损失函数更新模型的方式
#度量标准：用来判断模型性能
model.compile(optimizer=tf.train.AdagradOptimizer(learning_rate=0.0001),    #定义优化器
              loss='sparse_categorical_crossentropy',    #定义损失函数（在多分类的损失函数上，增加了稀疏性）
              metrics=['accuracy'])                 #定义评估模型性能的函数，这里使用的是模型的准确度


#训练模型
#1.提供数据
#2.学会关联图像和标签的关系
#3.对测试集进行预测


#调用model.fit方法
model.fit(train_images, train_labels, epochs=5)         #epochs为定义训练次数

#评估准确性
test_loss, test_acc = model.evaluate(test_images,test_labels)

print('Test accuracy:',test_acc)


#当测试集得到的精度和训练集之间得到的精度之间有差距，过度拟合是指机器学习模型在新数据上的表现比在训练数据上表现更差的现象

#作出预测(对批量图像而言)
predictions = model.predict(test_images)        #得到的结果是对每一个测试集的10个种类对应的百分比
print(predictions[0])
#百分比的最大值则为预测的结果
print(np.argmax(predictions[0]))


#对单个图像需要扩展维度
#获取图像(28,28)
img = test_images[0]
#扩展维度(1,28,28)
img = (np.expand_dims(img, 0))          #加入一个新轴，0为新轴的位置，即插入第一维

#然后预测图像，与上面的步骤一样
predictions = model.predict(img)
print(predictions)
prediction = predictions[0]
np.argmax(prediction)










