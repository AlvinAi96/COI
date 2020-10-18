'''
data_preprocess.py

该代码是将观点抽取的数据转换为BIO数据集(类似NER)
'''

import os
import pandas as pd
# from transformers import BertTokenizer
from tqdm import tqdm



data_path = './datasets/AE/' # 存放原始比赛官网下的原始数据集集
dataset_names = ['COTE-BD', 'COTE-MFW', 'COTE-DP']
# model_name_or_path='./prev_trained_model/bert-base'# 放下载好的中文分词预训练模型

# # 载入分词器
# tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)

def BIO_annotator(data_path, dataset_name, data_type):
    '''将原始数据转化为BIO标注类型的NER数据集
    参数：
        1. data_path:       (str): 原始数据集路径
        2. dataset_name:    (str): 对哪个数据集进行预处理
        3. data_type        (str): ‘train'或’test‘
    '''
    # 初始化写入文件
    save_path = './datasets/AE/' + dataset_name + '_new2/'
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    save_path = save_path + data_type + '.char.bmes'
    result_file = open(save_path, 'a', encoding='utf-8')

    # 导入数据
    if data_type == 'train':
        df = pd.read_csv(data_path + dataset_name + '/train.tsv', sep='\t', error_bad_lines=False)

        # 遍历每个样本做预处理
        ignore_num = 0
        print('处理数据集：%s - 数据类型：%s' % (dataset_name, data_type))
        for i in tqdm(range(len(df))):
            try:
                label = df['label'][i]
                text_a = df['text_a'][i].strip().replace(" ", "")
                index = -1
                index = text_a.find(label)
                # https://huggingface.co/transformers/glossary.html#input-ids
                # tokenized_sequence = tokenizer.tokenize(text_a)
                tokenized_sequence = [t for t in text_a]
                # 针对单例分词，初始化BIO标签
                label_list = ['O' for i in range(len(tokenized_sequence))]
                # 找到观点词并标注‘B’和‘I‘
                if index >= 0:
                    label_list[index] = 'B-NAME' # 观点词起点标注
                    for i in range(index+1, len(label)+index):
                        label_list[i] = 'I-NAME'

                for l in range(len(label_list)):
                    result_file.write("{0} {1}\n".format(tokenized_sequence[l], label_list[l]))

                result_file.write("\n") # 追加\n以便模型区分样本
            except:
                ignore_num += 1
                pass
        print("因中文分词器导致后面英文观点不能找到而忽略的评论数/评论总数：%s/%s" % (ignore_num, len(df)))

    elif data_type == 'test':
        df = pd.read_csv(data_path + dataset_name + '/test.tsv', sep='\t', error_bad_lines=False)
        print('处理数据集：%s - 数据类型：%s' % (dataset_name, data_type))
        for i in tqdm(range(len(df))):
            qid = df['qid'][i]
            text_a = df['text_a'][i].strip().replace(" ", "")
            # tokenized_sequence = tokenizer.tokenize(text_a)
            tokenized_sequence = [t for t in text_a]
            for l in range(len(tokenized_sequence)):
                # result_file.write("{}\n".format(tokenized_sequence[l]))
                result_file.write("{} O\n".format(tokenized_sequence[l]))
            result_file.write("\n") # 追加\n以便模型区分样本

    result_file.close()
    print("="*40)


for dataset_name in dataset_names:
    for data_type in ['train', 'test']:
        BIO_annotator(data_path, dataset_name, data_type)


