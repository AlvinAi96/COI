'''
post_process.py
该代码是用于后处理“句子级情感分析”和“观点级情感分析”的结果，然后输出符合提交格式的结果。
'''
import pandas as pd
import numpy as np
from scipy import stats
import os
import csv

task1_result_path = './model/roberta_wwm_large_5121_42/roberta_wwm_large_5121_42_'
task1_names = ['sub_CSC', 'sub_NS']

task2_result_path = './model/roberta_wwm_large_5121_42/task2_roberta_wwm_large_5121_42_'
task2_names = ['sub_SC', 'sub_SP']

# 创建最终结果保存文件夹
final_save_path = './submit/'
if os.path.exists(final_save_path) == False:
    os.makedirs(final_save_path)

def post_process(result_path, task_name, fold_num):
    result_path = result_path + str(fold_num) + '/' + task_name + '.csv'
    result_df = pd.read_csv(result_path)
    indexes = []
    predictions = []
    for i in range(len(result_df)):
        probs = result_df.iloc[i,1:]
        idx = result_df['qid'][i]
        label = np.argmax(probs)
        indexes.append(idx)
        predictions.append(label)
    results = pd.DataFrame({'index':indexes, 'prediction':predictions})
    return results

# 后处理句子级情感分析
for task1_name in task1_names:
    # 后处理结果
    result_dfset = []
    for k in range(5):
        results = post_process(task1_result_path, task1_name, k)
        result_dfset.append(results)

    # 依次为每个样本投票选出得票最高的预测结果
    final_indexes = []
    final_predictions = []
    for i in range(len(results)):
        tmp_preds = []
        for j in range(len(result_dfset)):
            tmp_preds.append(result_dfset[j]['prediction'][i])
        final_predictions.append(stats.mode(tmp_preds)[0][0])
        final_indexes.append(results['index'][i])
    final_result = pd.DataFrame({'index':final_indexes, 'prediction':final_predictions})

    # 保存结果
    if task1_name == 'sub_CSC':
        tsv_save_path = final_save_path + 'ChnSentiCorp.tsv'
        csv_save_path = final_save_path + 'ChnSentiCorp.csv'
    elif task1_name == 'sub_NS':
        tsv_save_path = final_save_path + 'NLPCC14-SC.tsv'
        csv_save_path = final_save_path + 'NLPCC14-SC.csv'
    elif task1_name == 'sub_SC':
        tsv_save_path = final_save_path + 'SE-ABSA16_CAME.tsv'
        csv_save_path = final_save_path + 'SE-ABSA16_CAME.csv'
    elif task1_name == 'sub_SP':
        tsv_save_path = final_save_path + 'SE-ABSA16_PHNS.tsv'
        csv_save_path = final_save_path + 'SE-ABSA16_PHNS.csv'
    final_result.to_csv(csv_save_path, index=False)
    with open(csv_save_path) as f:
        data = f.read().replace(',', '\t')
    with open(tsv_save_path,'w') as f1:
        f1.write(data)

# 后处理观点级情感分析
for task2_name in task2_names:
    # 后处理结果
    result_dfset = []
    for k in range(5):
        results = post_process(task2_result_path, task2_name, k)
        result_dfset.append(results)

    # 依次为每个样本投票选出得票最高的预测结果
    final_indexes = []
    final_predictions = []
    for i in range(len(results)):
        tmp_preds = []
        for j in range(len(result_dfset)):
            tmp_preds.append(result_dfset[j]['prediction'][i])
        final_predictions.append(stats.mode(tmp_preds)[0][0])
        final_indexes.append(results['index'][i])
    final_result = pd.DataFrame({'index':final_indexes, 'prediction':final_predictions})

    # 保存结果
    if task2_name == 'sub_CSC':
        tsv_save_path = final_save_path + 'ChnSentiCorp.tsv'
        csv_save_path = final_save_path + 'ChnSentiCorp.csv'
    elif task2_name == 'sub_NS':
        tsv_save_path = final_save_path + 'NLPCC14-SC.tsv'
        csv_save_path = final_save_path + 'NLPCC14-SC.csv'
    elif task2_name == 'sub_SC':
        tsv_save_path = final_save_path + 'SE-ABSA16_CAME.tsv'
        csv_save_path = final_save_path + 'SE-ABSA16_CAME.csv'
    elif task2_name == 'sub_SP':
        tsv_save_path = final_save_path + 'SE-ABSA16_PHNS.tsv'
        csv_save_path = final_save_path + 'SE-ABSA16_PHNS.csv'
    final_result.to_csv(csv_save_path, index=False)
    with open(csv_save_path) as f:
        data = f.read().replace(',', '\t')
    with open(tsv_save_path,'w') as f1:
        f1.write(data)


