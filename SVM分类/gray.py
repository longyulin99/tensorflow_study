#此程序用来实现灰度共生矩阵
# -*- coding: utf-8 -*-
import tensorflow as tf

import os
import cv2
import numpy as np
from skimage.feature import greycomatrix as gm
from skimage.feature import greycoprops as gp
import random as rd
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.preprocessing import  StandardScaler

from sklearn import svm


#灰度共生矩阵的实现
# 读取原来的图片，然后生成矩阵,file是图片的路径
def gray(file,d_x,d_y,gray_level=16):
    img=cv2.imread(file)

    #cv2.imshow("test",img)
    #cv2.waitKey()
    print("读取图片成功")
    #要判断图片的通道，然后进行
    if len(img.shape) == 3 or len(img.shape) == 4:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    #cv2.imshow("gray",img_gray)
    #cv2.waitKey()
    height=img_gray.shape[0]
    width=img_gray.shape[1]
    max_gray=img_gray.max()+1
    img_gray=img_gray*gray_level/max_gray
    ret=np.zeros([gray_level,gray_level])
    for j in range(height-d_y):
        for i in range(width-d_x):
            rows=img_gray[j][i]
            cols=img_gray[j+d_y][i+d_x]
            #这里将灰度分成了16个等级，然后需要将坐标一一对应起来
            ret_rows=int(rows*16)
            ret_cols=int(cols*16)
            ret[ret_rows][ret_cols]+=1.0

    ret=ret/float(img_gray.shape[0]*img_gray.shape[1])
    print('灰度共生矩阵实现成功')
    #将图片保存在本地的文件夹中
    parent_path=os.path.dirname(file)
    name=os.path.basename(file)
    #print(name)
    np.savetxt(parent_path+name+'_gray.csv',ret,delimiter=',')
    #print("成功")
    return ret


#测试调用
#file="C:\\J01_2018.06.17 13_22_33.jpg"
#gray(file,0,1,gray_level=16)


#实现将文件中所有的图片都转换为灰度共生矩阵并保存到另外一个文件夹中

















#定义获取矩阵特征的函数

def get_feature(file):
    feature = []
    im = cv2.imread(file,0)
    if len(im.shape) == 3 or len(im.shape) == 4:
        im= cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        im = im
    m = gm(im, [5], [0, np.pi / 2], levels=256, symmetric=True)
    #parent_path = os.path.dirname(file)
    #name = os.path.basename(file)
    # print(name)
    #np.savetxt(parent_path + name + '_gray.csv', m, delimiter=',')
    #print(m)
    con = gp(m, 'contrast')
    feature.append(con[0, 0])
    feature.append(con[0, 1])

    dis = gp(m, 'dissimilarity')
    feature.append(dis[0, 0])
    # feat
    feature.append(dis[0, 1])

    hom = gp(m, 'homogeneity')
    feature.append(hom[0, 0])
    feature.append(hom[0, 1])

    '''asm = gp(m,'ASM')
    feature.append(asm[0,0])
    feature.append(asm[0,1])'''

    df = np.array(feature)

    return df

#测试获取矩阵特征的函数
#file="C:\\J01_2018.06.17 13_22_33.jpg"
#df=get_feature(file)





#定义用于读取训练集和测试集的函数，函数返回记录图片顺序的列表
def get_xy(file, savex, savey):
    #此函数用来该转化灰度共生矩阵并提取特征值
    pic_l = os.listdir(file)
    rd.shuffle(pic_l)
    count = 0
    x = []
    y = []
    xx = []
    for pic in pic_l:
        print('getting')
        print(pic)
        if pic[-3:] == 'jpg':

            x_raw = get_feature(file + '\\' + pic)
            #x为灰度共生矩阵对应的特征，y是对应的类别
            x.append(x_raw)
            #获取对应的标签

            y.append(pic[-5])
            x = np.array(x)
            y = np.array(y)
            xtr_df = pd.DataFrame(x)
            xtr_df.to_csv('C:\\homework\\' + savex, mode='a', header=None, index=None)
            ytr_df = pd.DataFrame(y)
            ytr_df.to_csv('C:\\homework\\' + savey, mode='a', header=None, index=None)
            count += 1
            x = []
            y = []
            print(count)

    xx = np.array(x)
    yy = np.array(y)

    return pic_l

#使用函数getxy得到
#file='C:\\homework\\tianchiimage_for_class\\bianzhenyan'
#get_xy(file,'x_train.csv','y_train.csv')


def path_to_Uni(file):
    #该方法用于将文件夹中的中文路径改成对应的序号，并将映射关系保存到cvs文件中
    x=[]
    class_set=os.listdir(file)
    for class_num in range(len(class_set)):
        x.append([class_num,class_set[class_num]])
        os.rename(file+'\\'+class_set[class_num],file+'\\'+str(class_num))
    #将对应关系保存到homework文件中
    x=np.array(x)
    x=pd.DataFrame(x)
    x.to_csv('C:\\homework\\tianchi\\class_num.csv',encoding='utf-8-sig')

#调用方法
#file='C:\\homework\\tianchiimage_for_class'
#path_to_Uni(file)

def getxy(file,savex,savey):
    #此函数用来将数据集形成的灰度共生矩阵并保存到一个文件夹中
    x=[]
    y=[]
    #读取文件夹
    class_file=os.listdir(file)
    #遍历所有类别的文件夹
    for class_num in range(len(class_file)):
        #获取文件夹的名称并且保存到相应的位置上
        class_name=class_file[class_num]
        print("转换%s成功" % (class_name))
        #遍历图片
        img_set=glob(file+'\\'+class_name+ '\\*.jpg')
        for img_num in range(len(img_set)):
            #将图片转换为灰度共生矩阵并加入到x中
            img_gray=get_feature(img_set[img_num])
            x.append(img_gray)
            #x=np.array(x)
            #获取图片的标签并加入到y中
            y.append(class_name)
            #y=np.array(y)

    #最后将结果保存到文件夹中
    x=pd.DataFrame(x)
    y=pd.DataFrame(y)
    x.to_csv('C:\\homework\\tianchi\\' + savex, mode='a', header=None, index=None)
    y.to_csv('C:\\homework\\tianchi\\' + savey, mode='a', header=None, index=None)


#测试函数getxy
#file='C:\\homework\\tianchiimage_for_class'
#getxy(file,'x_train.csv','y_train.csv')


#此函数将test_a中的数据形成灰度共生矩阵保存在一个文件夹中
def get_test_x(file,save_gray,save_test_name):
    img_feature=[]
    #用于存放test的编号和名称
    img_test_name=[]
    img_set=os.listdir(file)
    for img_num in range(len(img_set)):
        #获取灰度共生矩阵的特征并保存到img_feature中
        img_feature.append(get_feature(file+'\\'+img_set[img_num]))
        img_test_name.append(img_set[img_num])
    #将img_feature保存到csv文件中
    img_feature=pd.DataFrame(img_feature)
    img_test_name=pd.DataFrame(img_test_name)
    img_feature.to_csv('C:\\homework\\tianchi\\' + save_gray, mode='a', header=None, index=None)
    img_test_name.to_csv('C:\\homework\\tianchi\\' + save_test_name, mode='a', header=None, index=None)










#使用SVM进行分类
'''
l1 = get_xy('D:\\1.3\\40s','x_train.csv','y_train.csv')
l2 = get_xy('D:\\1.3\\40s\\test','x_test.csv','y_test.csv')
'''
#读取训练集和测试集
x_train = np.array(pd.read_csv('C:\\homework\\tianchi\\x_train.csv',header = None))
y_train = np.array(pd.read_csv('C:\\homework\\tianchi\\y_train.csv',header = None))
'''
x_test = np.array(pd.read_csv('D:\\1.3\\40s\\x_test.csv',header = None))
y_test = np.array(pd.read_csv('D:\\1.3\\40s\\y_test.csv',header = None))
'''

#预处理数据
x_train=StandardScaler().fit_transform(x_train)

#使用交叉验证法进行整理数据
X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=0)





#训练
clf = svm.SVC(kernel = 'linear',C=0.5,degree = 4,probability=True)
clf.fit(X_train, y_train.ravel())




#得到结果
print(clf.score(X_train,y_train))
print(clf.score(X_test,y_test))


#使用该模型进行预测
# file='C:\\homework\\xuelang_round1_test_a_20180709'
# get_test_x(file,'test_gray.csv','test_img_name.csv')
#预测得到的结果保存到csv文件中
# x_test=np.array(pd.read_csv('C:\\homework\\tianchi\\test_gray.csv',header = None))
# x_test=StandardScaler().fit_transform(x_test)
#
# y_test=clf.predict(x_test)
# prob=clf.predict_proba(x_test)
# #获取概率的最大值
# prob_max=[]
# for i in range(len(prob)):
#     prob_max.append(max(prob[i]))
#
# print(y_test)
#
# print(prob_max)
# print('完毕')
# #将文件的名称和可能性拼接成一个完整的csv文件
# img_test_name=np.array(pd.read_csv('C:\\homework\\tianchi\\test_img_name.csv',header = None))
# #获取两者的shape
# print(img_test_name.shape)
# prob_max=np.array(prob_max)
# prob_max=prob_max.reshape(662,1)
#
# print(prob_max.shape)
#
# result=np.hstack([img_test_name,prob_max])
# head=['filename','probability']
# #result=np.vstack(head,result)
# result=pd.DataFrame(result)
# result.to_csv('C:\\homework\\tianchi\\result.csv', mode='a', header=head, index=None)
# print('结果输出完毕')









