## Comment of Interest (COI)
**创建人**：艾宏峰<br>
**创建人**：2020-08-15<br>

COI，是Comment of Interest的缩写，如果熟悉美剧《疑犯追踪》(英文名：Person of Interest， POI)的朋友应该会对我取得代号有熟悉的感知。如英文名所表明的那样，我起初是希望能构建并实现一个针对网络评论文本数据进行数据挖掘的项目(包括并不限于：爬虫、文本处理、情感分析和與情分析等等)，但最后还是没太做的太深，只是实现了一个项目的雏形吧，姑且称它为COI Demo。它目前能实现的功能如下：

1. 电商评论爬虫（Web Crawler）
2. 句子级情感分析（Sentence-level Sentiment Classification）
3. 观点级情感分析（Aspect-level Sentiment Classification）
4. 观点抽取（Opinion Target Extraction）

详细项目解释请见：[练手项目：电商评论文本挖掘 - COI (Comment of interest)](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247483885&idx=1&sn=ffd40a38193b9d846c5eca8ae8e8744d&chksm=fa041f86cd739690f3df40516b406835f8ff44ae51224edccec4f5238cd4d230de50d5e51985&token=370538825&lang=zh_CN#rd)

### 环境配置
由于观点抽取、句子级情感分析和观点级情感分析需要基于huggingface等人开源维护的NLP代码库[Transformers](https://github.com/huggingface/transformers)，所以需要配置相应的环境以便顺利运行代码。
构建虚拟环境并安装Transformers的详细教程，请参考[Transformers Installation](https://github.com/huggingface/transformers)
```
# 创建虚拟环境
conda create -n transformer_env python=3.7
conda activate transformer_env

# 由于我CUDA为10.0，CUDNN为7.5.1，因此我下载的是pytorch 1.2.0
#（注意Transformers只支持Python 3.6+, PyTorch 1.0.0+，TensorFlow 2.0）
# pytorch下载命令查询地址：https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# 安装Transformers
pip install transformers

# 安装其他库（自行按需安装即可）
pip install boto3 pandas tqdm scikit-learn requests urllib3 pillow sacremoses sentencepiece

```

### 一、电商评论爬虫
具体代码实现的解释请见博文：[使用Selenium爬取京东电商数据(以手机商品为例)](https://www.cnblogs.com/alvinai/p/13545727.html)
```
cd crawlers
# 依次爬取京东的列表页和详情页
python overview_JDcrawler.py
python detail_JDcrawler
```

### 二、句子级情感分析
本来应该是要我人工标注上述爬取的文本评论，但是后面发现了百度AI Studio开放了情感分析主体的[千言数据集](https://aistudio.baidu.com/aistudio/competition/detail/50)，并以比赛的形式(即类似练习赛)鼓励参赛者用里面的数据集进行练习。开放的千言数据集涵盖了包括句子级情感分类（Sentence-level Sentiment Classification）、评价对象级情感分类（Aspect-level Sentiment Classification）、观点抽取（Opinion Target Extraction）三个经典任务。所以我就基于他们标注好的数据集进行模型训练。

#### 1. 数据预处理
下载千言数据集压缩包并放统一置在``./SA/``文件夹下。
```
# 批量将data/baidu_data/内放置的数据集压缩包进行批量解压
cd SA/data/baidu_data
ls *.zip | xargs -n1 unzip -o -P infected

# 开始数据预处理
cd ..
python data_preprocess_task1.py
```
#### 2. 模型训练、验证及预测
下载[中文RoBERTa预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)，并将下载好的RoBERTa-wwm-ext-large模型放在``SA/model/chinese_roberta_wwm_large_ext_pytorch``下，并将配置文件``bert_config.json``改名为``config.json``，然后可以执行下面的命令了。
```
cd SA/source
# 正式训练模型前，run_bert_2562_task1.py内的data_dir和output_dir的后缀要更改(对应第k折数据及结果)
# 这里可以用我的默认设置即可
python run_bert_2562_task1.sh
```
最后5折交叉训练后的结果在``SA/model/roberta_wwm_large_5121_42/``下。

### 三、观点级情感分析
代码过程与上面的句子级情感分析类似。
#### 1. 数据预处理
```
# 开始数据预处理
cd SA/data
python data_preprocess_task2.py
```
#### 2. 模型训练、验证及预测
```
cd SA/source
python run_bert_2562_task2.sh
```
#### 3. 结果后处理
这里统一处理句子级情感分析和观点级情感分析的结果，分别对它们各自的5折结果进行投票，最后后处理成符合提交格式的结果文件。
```
python source/post_process.py
```

### 四、观点抽取
观点抽取是用算法从指定文本中抽取处理文本的评价对象，例如：OPPO FIND X2是一部很好的旗舰级手机。那么观点抽取算法会得到：OPPO FIND X2。
#### 1. 数据预处理
观点抽取本质上就是命名实体识别NER任务，只不过不需要很复杂BIO去识别出实体的类型，如地点机构人物，只用简单的BIO就行了。所以要先对数据进行预处理成以BIO标注格式的数据。但首先也要先将之前我们下载好的RoBERTa-wwm-ext-large模型文件夹内的文件放在``NER/prev_trained_model/bert-base/``下，而且还要把之前解压出的COTE-BD、COTE-DP和COTE-MFW数据文件夹放置在``NER/datasets/AE/``下面，然后再进行数据预处理：
```
cd NER
cd python data_preprocess_single_word.py
```
最后``NER/datasets/AE/``下会出现以“_new2”为结尾的新数据集，都是预处理好的。

#### 2. 模型训练、验证及预测
此时，依次将不同数据集放入cner分别训练、验证及预测。因为COTE没有验证集，所以暂由train代替dev。因此在``run_ner_softmax.py``中第203行改为train。
```
# 将AE/datasets/COTE-BD_new2下的文件复制到AE/datasets/cner/下后，运行下列命令
# 注意output_dir要随着数据集而更改
python run_ner_softmax.py \
  --model_type=bert \
  --model_name_or_path=./prev_trained_model/bert-base \
  --task_name="cner" \
  --do_train \
  --do_eval \
  --do_predict \
  --do_lower_case \
  --loss_type=ce \
  --data_dir=./datasets/cner/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=128 \
  --per_gpu_train_batch_size=8 \
  --per_gpu_eval_batch_size=8 \
  --learning_rate=3e-5 \
  --num_train_epochs=3.0 \
  --logging_steps=448 \
  --save_steps=448 \
  --output_dir=./outputs/cner_cotebd_output/ \
  --overwrite_output_dir \
  --seed=42
```
训练且预测好后，``NER/outputs/``下会出现三个数据集的模型结果文件夹。
#### 3. 结果后处理
```
cd NER
python post_process.py
```

最后将后处理后的结果打包提交给[千言数据集：情感分析](https://aistudio.baidu.com/aistudio/competition/detail/50)，就能得到以下的成绩：
|Score|NLPCC14-SC|ChnSentiCorp|COTE_BD|COTE_DP|SE-ABSA16_CAME|COTE_MFW|SE-ABSA16_PHNS|提交时间|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Alvin0310的团队|0.8282|0.8384|0.9392|0.8729|0.893|0.7208|0.8901|0.6427|2020-10-17 21:14|
