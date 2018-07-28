import tensorflow as tf

#导入数据集
import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)



#创建一个接受数据的对象x，希望接受任意数量的数据，故前一维使用None，可以为接收的图像的数据
x=tf.placeholder("float",[None,784])

#创建两个变量w(权重的矩阵)，以及b（每个label对应的偏移量）,使用Variables表示在计算的过程中是一个可变的量
w=tf.Variable(tf.zeros[784,10])
b=tf.Variable(tf.zeros[10])

#创建模型,softmax为激活函数，可以得到不同的图像对应的数字的概率
y=tf.nn.softmax(tf.matmul(x,w)+b)



#训练模型
#指定一个指标表示这个模型是坏的，称为成本或者损失loss，最小化值
#使用交叉熵
#首先需要一个量接收正确的数据,同样与x相同，指定不定的第一维
y_=tf.placeholder("float",[None,10])

#然后计算交叉熵,reduce_sum为求和
cross_entropy=-tf.reduce_sum(y_*tf.log(y))

#使用优化算法不断修改变量来降低成本,使用了梯度下降算法，以0.01的学习速率最小化交叉熵
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#建立完模型后，需要用一个量来初始化创建的变量
init=tf.initialize_all_variables()

#启动模型，并且在模型中初始化变量
sess=tf.Session()
sess.run(init)

#开始训练模型，让模型循环训练1000次
#大致的意思是，batch_xs,batch_ys用来接收训练集中的随机100条数据，然后将数据传入x与y_中，然后运行train_step来最小化成本
for i in range(1000)
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})


#评估模型,argmax可以得到某个tensord对象在某一维度上其数据最大值所在的索引值，后面的1代表维度，equal来预测是否匹配
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#得到的是布尔类型的数组，需要转换类型到浮点数然后取平均值,tf.cast()为转换类型的函数，reduce_mean用来取平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#与测试数据相比对，得到一个概率值
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})









