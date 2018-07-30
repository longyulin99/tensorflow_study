import os
from glob import glob
import shutil
#此文件将数据进行分类

if __name__ == '__main__':

    data_dir = 'C:\\homework\\tianchiimage_cut\\'

    train_dirs = ['xuelang_round1_train_part1_20180628', 'xuelang_round1_train_part2_20180705', 'xuelang_round1_train_part3_20180709']

    class_dict = {}
    class_idx = 0
    #需要创建文件夹保存整理的文件
    for train_dir in train_dirs:
        train_path = data_dir + train_dir + '\\'
        class_names = os.listdir(train_path)
        # print(class_paths)
        for class_name in class_names:
            img_for_a_class=glob(train_path + class_name + '\\*.jpg')   #返回的是路径的列表
            #print(img_for_a_class)
            img_num_in_class = len(img_for_a_class)
            if class_name not in class_dict:
                class_dict[class_name] = img_num_in_class
                class_idx += 1
                #创建目录
                path="C:\\homework\\tianchiimage_for_class\\"+class_name
                #先判断文件是否存在
                if os.path.exists(path)==False:
                    os.mkdir(path)
                #拷贝文件
                for img_num in range(img_num_in_class):
                    #获取路径的名称
                    img_name=os.path.basename(img_for_a_class[img_num])
                    shutil.copy(img_for_a_class[img_num],path+'\\'+img_name)
            else:
                #当文件中已经存在时
                class_dict[class_name] += img_num_in_class
                #在当前目录下找到对应序号的文件然后进行拷贝
                path="C:\\homework\\tianchiimage_for_class"
                new_file=os.listdir(path)
                for file_num in range(len(new_file)):
                    if new_file[file_num]==class_name:
                        #找到对应文件的序号，遍历图片并拷贝到指定的目录中
                        for img_num in range(img_num_in_class):
                            img_name=os.path.basename(img_for_a_class[img_num])
                            shutil.copy(img_for_a_class[img_num],path+'\\'+class_name+'\\'+img_name)
                        break
    total_img_num = 0
    for key in class_dict:
        print('class: %s, img num: %d' % (key, class_dict[key]))
        total_img_num += class_dict[key]
    print('total class num: %d' % class_idx)
    print('total image num: %d' % total_img_num)


    #将文件分成训练集和测试集
    #思路重新创建一个文件夹作为测试集的文件，然后在刚刚新生成文件中的每个子文件进行分层抽样

#result output
# class: 吊纬, img num: 6
# class: 扎洞, img num: 48
# class: 正常, img num: 1316
# class: 毛斑, img num: 35
# class: 修印, img num: 1
# class: 剪洞, img num: 5
# class: 厚薄段, img num: 1
# class: 吊弓, img num: 3
# class: 吊经, img num: 135
# class: 回边, img num: 6
# class: 嵌结, img num: 8
# class: 弓纱, img num: 8
# class: 愣断, img num: 2
# class: 扎梳, img num: 3
# class: 扎纱, img num: 1
# class: 擦伤, img num: 2
# class: 擦毛, img num: 2
# class: 擦洞, img num: 123
# class: 楞断, img num: 4
# class: 毛洞, img num: 53
# class: 毛粒, img num: 1
# class: 污渍, img num: 12
# class: 油渍, img num: 13
# class: 破洞, img num: 4
# class: 破边, img num: 7
# class: 粗纱, img num: 4
# class: 紧纱, img num: 1
# class: 纬粗纱, img num: 1
# class: 线印, img num: 2
# class: 织入, img num: 4
# class: 织稀, img num: 53
# class: 经粗纱, img num: 2
# class: 经跳花, img num: 2
# class: 结洞, img num: 1
# class: 缺纬, img num: 14
# class: 缺经, img num: 43
# class: 耳朵, img num: 1
# class: 蒸呢印, img num: 2
# class: 跳花, img num: 58
# class: 边扎洞, img num: 16
# class: 边白印, img num: 1
# class: 边缺纬, img num: 2
# class: 边针眼, img num: 7
# class: 黄渍, img num: 5
# class: 厚段, img num: 1
# class: 夹码, img num: 1
# class: 明嵌线, img num: 1
# class: 边缺经, img num: 1
# total class num: 48
# total image num: 2022
