import os
from glob import glob
import  xml.dom.minidom
from PIL import Image
import shutil

#此程序用来裁剪图片，将图片中的瑕疵扣出来，放入到C:\\homework\\tianchiimage_no_cut中
#遍历文件中的每一张图片，然后保存在相应的文件夹中，第一张图片需要创建一个新的文件夹
if __name__ == '__main__':
    data_dir = 'C:\\大三各种作业\\天池大赛\\'

    train_dirs = ['xuelang_round1_train_part1_20180628', 'xuelang_round1_train_part2_20180705',
                  'xuelang_round1_train_part3_20180709']

    for i in range(len(train_dirs)):
        path=data_dir+train_dirs[i]
        #创建一个新的文件夹存放
        new_path='C:\\homework\\tianchiimage_no_cut\\'+train_dirs[i]
        #检测文件夹是否存在
        if os.path.exists(new_path)==False:
            #创建新的文件夹
            os.mkdir(new_path)

        #遍历文件下的每一个文件，然后创建一个文件
        class_set=os.listdir(path)
        for j in range(len(class_set)):
            #创建文件
            class_path=path+'\\'+class_set[j]
            new_class_path=new_path+'\\'+class_set[j]
            if os.path.exists(new_class_path)==False and class_set[j]!="正常":
                os.mkdir(new_class_path)
                # 遍历每个class中的图片，进行裁剪后保存到对应的文件夹中
                img_set = glob(class_path + '\\*.jpg')
                xml_set = glob(class_path + '\\*.xml')
                for k in range(len(img_set)):
                    # 读取对应的xml文件得到xmin,ymin,xmax,ymax坐标
                    dom = xml.dom.minidom.parse(xml_set[k])
                    root = dom.documentElement
                    # 获取对应的元素的值
                    xmin_element = root.getElementsByTagName("xmin")
                    xmin = int(xmin_element[0].firstChild.data)
                    xmax_element = root.getElementsByTagName("xmax")
                    xmax = int(xmax_element[0].firstChild.data)
                    ymin_element = root.getElementsByTagName("ymin")
                    ymin = int(ymin_element[0].firstChild.data)
                    ymax_element = root.getElementsByTagName("ymax")
                    ymax = int(ymax_element[0].firstChild.data)
                    print("xmin:%s,ymin:%s,xmax:%s,ymax:%s" % (xmin, ymin, xmax, ymax))
                    # 将图片按照指定的坐标进行裁剪
                    img = Image.open(img_set[k])
                    img_cut = img.crop((xmin, ymin, xmax, ymax))
                    # 保存图片
                    # 获取图片的名称
                    img_name = os.path.basename(img_set[k])
                    img_cut.save(new_class_path + '\\' + img_name)
            elif class_set[j]=='正常'and os.path.exists(new_class_path)==False:
                #直接将文件复制过去
                shutil.copytree(class_path,new_class_path)





