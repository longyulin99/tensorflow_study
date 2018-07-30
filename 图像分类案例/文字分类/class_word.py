#此模型用于文字分类，将电影的评论分成是正面的还是负面的
#样本说明：将样本让你分成25000条训练数据以及25000条测试数据
#训练集和测试集其中包含的正负面评论是相同的

import tensorflow as tf
from tensorflow import keras
import numpy as np


#IMDB数据集已经被预处理，其中单词序列已经被整理成整数序列，整数表示字典中对应的单词
#