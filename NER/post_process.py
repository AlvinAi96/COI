'''
post_process.py
该代码是用于后处理“观点抽取”的结果，然后输出符合提交格式的结果。
'''

import pandas as pd
import os
import re,string

punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'

data_types = ['bd', 'mfw', 'dp']

# 创建最终结果保存文件夹
final_save_path = './submit/'
if os.path.exists(final_save_path) == False:
    os.makedirs(final_save_path)

for data_type in data_types:
    print("data_type", data_type)
    # 读取预测结果
    output_path = './outputs/cner_cote{}_output/bert/test_prediction.json'.format(data_type)
    output_df = pd.read_json(output_path, encoding='utf-8', lines=True)

    # 重整预测结果
    # 读取分词结果
    f = open('./datasets/AE/COTE-{}_new2/test.char.bmes'.format(data_type.upper()), "r")
    lines = f.readlines()
    sample_count = 0
    indexes = [] # 存放例子id
    words = [] # 存放例子分词
    tmp_words = []
    for line_no, line in enumerate(lines):
        if 'O\n' in line:
            word = line[:-3]
            tmp_words.append(word)
        else:
            words.append(tmp_words)
            indexes.append(sample_count)
            sample_count += 1
            tmp_words = []

    # 为每个预测结果中根据预测观点位置，提取出预测观点词
    # 源码不知道为什么不对第一个样例预测，这里暂时补为‘无’
    final_indexes = [0]
    final_predictions = ['无']
    miss_pred_count = 0
    for i in range(len(output_df)):
        # 如果有预测结果就正常抽取，若没就抽前三个即可
        try:
            start_idx = output_df['entities'][i][0][1]
            end_idx = output_df['entities'][i][0][2]+1
            aspect_word = ''.join(words[i+1][start_idx:end_idx])
            aspect_word = re.sub(r"[%s]+" %punc, "", aspect_word)
        except:
            miss_pred_count+=1
            aspect_word = '无'
        # aspect_word = aspect_word.replace("[UNK]", "")
        # aspect_word = re.sub(r"[%s]+" %punc, "", aspect_word)
        index = indexes[i]+1
        final_indexes.append(index)
        final_predictions.append(aspect_word)

    # 保存结果
    final_result = pd.DataFrame({'index':final_indexes, 'prediction':final_predictions})
    csv_save_path = './submit/COTE_{}.csv'.format(data_type.upper())
    tsv_save_path = './submit/COTE_{}.tsv'.format(data_type.upper())
    final_result.to_csv(csv_save_path, index=False)
    with open(csv_save_path) as f:
        data = f.read().replace(',', '\t')
    with open(tsv_save_path,'w') as f1:
        f1.write(data)
    print("有%s条样本没预测" % miss_pred_count)
    print("="*20)
