import tensorflow as tf

#导入数据集
import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)



x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])
#定义两个函数来初始化变量
#初始化权重
#shape为一个1*4的向量，[batch(训练时一个batch的图像数量), in_height(高), in_width（宽）, in_channels（有多少通道）]，
def weight_variable(shape):
    #truncated_normal为正态分布，没有指定均值，标准差为0.1
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

#初始化偏差
def bias_variable(shape):
    #因为使用的是ReLU神经元，先用一个较小的正数来初始化偏差
    #使用constant赋值，则是不可改变的张量
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#卷积与池化
#使用步长为1，0边距的模板，可保证输入和输出是同一个大小
#x为输入图像，W为卷积核，tf.nn.conv2d定义卷积函数
def conx2d(x,W):
    #即使用水平步长和垂直步长为1，当卷积核移动到超过图像边界的使用填充0，因为padding='SAME'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#定义池化函数,这里使用的是最大值池化
def max_pool_2x2(x):
    #ksize为池化窗口的大小，为2*2，因为不在batch和channel上做改动，故设置成1,strides为步长，返回一个Tensor
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


#实现第一层卷积（一个卷积由一个卷积接一个最大值池化组成），[5,5,1,32]前面两个是大小，1为输入的通道数目，32为输出的通道数目
W_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])

#将图像x变成4D向量，第二维第三维为图像的长和宽，最后一维为图像的颜色通道数，第一维为图像的数量，等于-1时则不指定数量
x_image=tf.reshape(x,[-1,28,28,1])

#将x_image和权值向量进行卷积，加上偏置项，然后使用ReLU激活函数，然后池化
h_conv1=tf.nn.relu(conx2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)

#第二层卷积，第二层，每5*5个patch会得到64个特征
W_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])

h_conv2=tf.nn.relu(conx2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#密集连接层，图像已经减少到7*7，然后加入一个有1024个神经元的全连接层(将图像所有的像素连接在一起)，用于处理整个图片
#对每一个像素有对应的b
W_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

#将池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏移量，然后对其使用ReLU激活函数
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

#dropout，为了减少过拟合，在输出层之前加入dropout,用一个placeholder来表示一个神经元的输出在dropout中保持不变的概率
keep_prob=tf.placeholder("float")
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
#????dropout原理

#然后添加一个softmax层，？？？为什么是1024
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#训练和评估模型
#同样定义交叉熵,使用更加复杂的ADAM优化器来做梯度最速下降
cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#对比评估的结果与实际值,计算平均值
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#启动模型
sess=tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch=mnist.train.next_betch(50)
    if i%100==0:
        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})#????
        print "step %d,training accuracy %g"%(i,train_accuracy)
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})









