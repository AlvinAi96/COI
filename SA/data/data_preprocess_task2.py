'''
data_preprocess_task2.py
该代码文件主要是用于数据的预处理(评价对象级情感分类)。
创建者：艾宏峰
创建日期：20201007
修改日期：20201007


代码参考：
001_初赛复赛数据集合并.ipynb
002_原始数据分层抽样划分5折.ipynb
004-数据预处理（替换英文字符).ipynb
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import random
import re

# ====================
# 1. 数据集导入及合并
# ====================
print("开始数据集导入及合并...")
# 导入SE-ABSA16_PHNS和SE-ABSA16_CAME数据集并合并它们
SP_train = pd.read_csv("baidu_data/SE-ABSA16_PHNS/train.tsv", sep='\t')
SP_test = pd.read_csv("baidu_data/SE-ABSA16_PHNS/test.tsv", sep='\t')
SC_train = pd.read_csv("baidu_data/SE-ABSA16_CAME/train.tsv", sep='\t')
SC_test = pd.read_csv("baidu_data/SE-ABSA16_CAME/test.tsv", sep='\t')

# 合并训练用的数据集(合并后20346个样本)
train = pd.concat([SP_train, SC_train], axis = 0).reset_index(drop = True)

# 清除label是空值的数据
train['label'] = train['label'].fillna(-1)
train['label'] = train[train['label']!=-1]
train['label'] = train['label'].astype(int)

# 填补评论的空值
train['text_a'] = train['text_a'].fillna('无')
SP_test['text_a'] = SP_test['text_a'].fillna('无')
SP_test['text_b'] = SP_test['text_b'].fillna('无')
SC_test['text_a'] = SC_test['text_a'].fillna('无')
SC_test['text_b'] = SC_test['text_b'].fillna('无')
SP_test['label'] = 0
SC_test['label'] = 0

print("SE-ABSA16_PHNS和SE-ABSA16_CAME数据集合并后样本数：%d" % len(train))
print("数据集导入及合并完成！")
print("="*40)



# ====================
# 2. 替换英文字符
# ====================
print("开始替换英文字符...")

def replace_punctuation(example):
    '''替换英文字符'''
    example = list(example)
    pre = ''
    cur = ''
    for i in range(len(example)):
        if i == 0:
            pre = example[i]
            continue
        pre = example[i-1]
        cur = example[i]
        # [\u4e00-\u9fa5] 中文字符
        if re.match("[\u4e00-\u9fa5]", pre):
            if re.match("[\u4e00-\u9fa5]", cur):
                continue
            elif cur == ',':
                example[i] = '，'
            elif cur == '.':
                example[i] = '。'
            elif cur == '?':
                example[i] = '？'
            elif cur == ':':
                example[i] = '：'
            elif cur == ';':
                example[i] = '；'
            elif cur == '!':
                example[i] = '！'
            elif cur == '"':
                example[i] = '”'
            elif cur == "'":
                example[i] = "’"
    return ''.join(example)

# 替换英文字符 - 训练集
rep_train_a = train['text_a'].map(replace_punctuation)
rep_train_b = train['text_b'].map(replace_punctuation)
rep_train = pd.concat([train['label'], rep_train_a, rep_train_b], axis = 1)
# 替换英文字符 - SP测试集
rep_SP_test_a = SP_test['text_a'].map(replace_punctuation)
rep_SP_test_b = SP_test['text_b'].map(replace_punctuation)
rep_SP_test = pd.concat([SP_test[['qid', 'label']], rep_SP_test_a, rep_SP_test_b], axis = 1)
# 替换英文字符 - SC测试集
rep_SC_test_a = SC_test['text_a'].map(replace_punctuation)
rep_SC_test_b = SC_test['text_b'].map(replace_punctuation)
rep_SC_test = pd.concat([SC_test[['qid', 'label']], rep_SC_test_a, rep_SC_test_b], axis = 1)

train = rep_train
SP_test = rep_SP_test
SC_test = rep_SC_test
del rep_train, rep_SP_test, rep_SC_test
print("替换英文字符完成！")
print("="*40)



# ====================
# 3. 5折分层抽样
# ====================
print("开始5折分层抽样...")
X = np.array(train.index)
y = train.loc[:, 'label'].to_numpy()

def generate_data(random_state = 2020):
    '''5折抽样'''
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = random_state)
    i = 0
    for train_index, dev_index in skf.split(X, y):
        print("Fold:", i, "| 训练集大小:", len(train_index), "| 验证集大小:", len(dev_index))
        # 定义抽样结果的保存路径
        data_save_dir = "./baidu_data/task2_data_StratifiedKFold_{}/data_origin_{}/".format(random_state, i)
        if not os.path.exists(data_save_dir):
            os.makedirs(data_save_dir)
        # 获取抽样结果并保存它们
        tmp_train = train.iloc[train_index]
        tmp_dev = train.iloc[dev_index]
        tmp_train.to_csv(data_save_dir + "train.csv")
        tmp_dev.to_csv(data_save_dir + "dev.csv")
        # 在每折文件夹内都放入测试集，方便后续预测
        SP_test.to_csv(data_save_dir + 'SP_test.csv')
        SC_test.to_csv(data_save_dir + 'SC_test.csv')
        i+=1

generate_data(random_state = 2020)
print("5折分层抽样完成！")