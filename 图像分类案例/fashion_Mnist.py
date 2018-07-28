#Tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

#帮助库
import numpy as np
#导入python中的2D绘图库
import matplotlib.pyplot as plt

print(tf.__version__)


#从mnist加载数据
fashion_mnist=keras.datasets.fashion_mnist

#60000个训练样本与10000个测试样本，是使用load_data()加载数据
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

#定义class，序号从0到10一一对应
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#输出训练样本的数量以及大小,可用shape属性得到
print(train_images,shape)

#对应有60000个标签
print(len(train_labels))

#将训练的标签进行输出
print(train_labels)

#同样可以得到测试数据的长度以及大小


#对图像进行预处理，使用matplotlib.pyplot库

#首先需要将图像的值转换称0~1之间的浮点数
train_images=train_images/255.0
test_images=test_images/255.0


#




