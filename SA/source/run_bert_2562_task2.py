# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import

import argparse
import csv
import logging
import os
import random
import sys
from io import open
import pandas as pd
import numpy as np
import torch
import gc
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import f1_score
import json
from pytorch_transformers.modeling_bert import BertForSequenceClassification, BertConfig
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.tokenization_bert import BertTokenizer
from itertools import cycle

# 初始化日志
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
# 收集Transformers我们需要的模型类
MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
#
ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in ( BertConfig,)), ())
# 获取logger对象
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid # 唯一标识id
        self.text_a = text_a # 评价对象
        self.text_b = text_b # 评价内容
        self.label = label # 情感极性标签


class InputFeatures(object):
    # 转为InputFeatures类
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for _, input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label

def read_examples(input_file, is_training):
    # 读取数据文件
    df=pd.read_csv(input_file)
    examples=[]
    for val in df[['Unnamed: 0', 'text_a', 'text_b', 'label']].values:
        examples.append(InputExample(guid=val[0],text_a=val[1],text_b=val[2],label=val[3]))
    return examples

def convert_examples_to_features(examples, tokenizer, max_seq_length,split_num,
                                 is_training):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # SWAG(Situations With Adversarial Generations)
    # 给出一个陈述句子和4个备选句子, 判断前者与后者中的哪一个最有逻辑的连续性, 相当于阅读理解问题.
    # [CLS] - 分离符，用来分割样本
    # [SEP] - 分隔符，用来分割样本内的不同句子
    # 举例（BERT输入的分词及片段id）：
    # 分词tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    # segement_ids：0     0  0    0    0     0       0 0     1  1  1  1   1 1
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(examples):
        # 对评论分词
        context_tokens=tokenizer.tokenize(example.text_a) # 对评论对象分词
        ending_tokens=tokenizer.tokenize(example.text_b) # 对评论内容分词

        # 将正文拆分为几份？这里我们split_num=1，即评论不会被分割
        skip_len=len(context_tokens)/split_num
        choices_features = []
        for i in range(split_num):
            context_tokens_choice=context_tokens[int(i*skip_len):int((i+1)*skip_len)]
            # 修剪输入文本以符合最大序列输入长度（例512）-3(3个标记符[CLS][SEP][SEP])，
            _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)
            # bert模型的输入：input_ids，input_mask，segment_ids
            # 获得分词(tokens)、片段id(segment_ids)、文本id(input_ids)和文本掩码(input_mask)
            # 记句子为S，属性词为A。称[CLS]+S+[SEP]+A+[SEP]
            tokens = ["[CLS]"]+ ending_tokens + ["[SEP]"] +context_tokens_choice  + ["[SEP]"]
            segment_ids = [0] * (len(ending_tokens) + 2) + [1] * (len(context_tokens_choice) + 1)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            # input_mask和最大长度有关，假设我们定义句子的最大长度是120，当前句子长度是100，那么input_mask前100个元素都是1，其余20个就是0
            input_mask = [1] * len(input_ids)

            # Padding处理
            padding_length = max_seq_length - len(input_ids) # 计算需要padding的长度
            input_ids += ([0] * padding_length) # [Pad] - 0
            input_mask += ([0] * padding_length) # 正常文本为1, padding为0
            segment_ids += ([0] * padding_length)
            choices_features.append((tokens, input_ids, input_mask, segment_ids))

            label = example.label
            # 只打印第一个输入样例
            if example_index < 1 and is_training:
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example_index))
                logger.info("guid: {}".format(example.guid))
                logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581','_')))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
                logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
                logger.info("label: {}".format(label))

        # 获取InputFeatures类
        features.append(
            InputFeatures(
                example_id=example.guid,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # 修剪序列以符合最大长度

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        # 现序列总长度=评论长度+[]长度
        total_length = len(tokens_a) + len(tokens_b)
        # 如果现序列总长度<=最大要求长度，则不修改
        # 如果现序列总长度>最大要求长度且评论长度>[]长度，则修剪正文长度
        # 如果现序列总长度>最大要求长度且评论长度<[]长度，则修剪标题长度
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    # 求Macro F1 score
    outputs = np.argmax(out, axis=1)
    return f1_score(labels,outputs,labels=[0,1],average='macro')

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

def set_seed(args):
    # 规定统一的种子数
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters(即required=True的参数必须在命令上出现)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="数据集路径. The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="模型类型(这里为bert). Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="下载好的预训练模型. Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--meta_path", default=None, type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型预测和断点文件的存放路径. The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="预训练的配置名字或路径. Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="预训练分词器名字或路径. Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="从亚马逊s3下载的预训练模型存放路径. Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="最长序列长度. The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="是否训练. Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                        help="是否测试. Whether to run testing.")
    parser.add_argument("--predict_eval", action='store_true',
                        help="是否预测验证集. Whether to predict eval set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否验证. Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="是否训练中跑验证. Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="是否用小写模型. Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="训练时每个GPU/CPU上的batch size. Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="验证时每个GPU/CPU上的batch size. Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="反向传播前梯度累计的次数. Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Adam的初始学习率. The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="权重衰减系数. Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam的Epsilon系数. Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help=" 如果所有参数的gradient组成的向量的L2 norm大于max norm，那么需要根据L2 norm/max_norm进行缩放。从而使得L2 norm小于预设的clip_norm. Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="训练epoch数. Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--lstm_hidden_size", default=300, type=int,
                        help="")
    parser.add_argument("--lstm_layers", default=2, type=int,
                        help="")
    parser.add_argument("--lstm_dropout", default=0.5, type=float,
                        help="")

    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--report_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="线性warmup的steps. Linear warmup over warmup_steps.")
    parser.add_argument("--split_num", default=3, type=int,
                        help="测试集划分. text split")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="日志更新steps. Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="断点文件保存steps. Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="评估所有的断点. Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="不用cuda. Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="重写输出路径. Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="重写训练和评估的缓存. Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="初始化用的随机种子. random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="是否用16位混合精度. Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="fp16的优化level. For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="为了分布式训练. For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="远程debug用的ip. For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="远程debug用的端口. For distant debugging.")
    parser.add_argument("--freeze", default=0, type=int, required=False,
                        help="冻结BERT. freeze bert.")
    parser.add_argument("--not_do_eval_steps", default=0.35, type=float,
                        help="not_do_eval_steps.")
    args = parser.parse_args()


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        # 如果无指定GPU或允许使用CUDA，就使用当前所有GPU
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # 指定使用哪个GPU（local_rank代表当前程序进程使用的GPU标号）
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging 初始化日志
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed 设置种子数
    set_seed(args)

    # 创建存放路径
    try:
        os.makedirs(args.output_dir)
    except:
        pass

    # 载入预训练好的BERT分词器
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)

    # 载入预设好的BERT配置文件
    config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=2)

    # Prepare model 载入并配置好基于BERT的序列分类模型
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,args,config=config)

    # 开启FP16
    if args.fp16:
        model.half()
    model.to(device)
    # 如果是指定了单个GPU，用DistributedDataParallel进行GPU训练
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # 如果有多个GPU，就直接用torch.nn.DataParallel，会自动调用当前可用的多个GPU
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # 总batch size = GPU数量 * 每个GPU上的batch size
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    if args.do_train:
        # Prepare data loader 导入数据并准备符合格式的输入
        train_examples = read_examples(os.path.join(args.data_dir, 'train.csv'), is_training = True)
        train_features = convert_examples_to_features(
            train_examples, tokenizer, args.max_seq_length, args.split_num, True)
        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
        # 如果无指定GPU就随机采样，如果指定了GPU就分布式采样
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # 准备dataloader
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps)
        # 训练steps
        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer 准备优化器
        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer]

        # no_dacay内的参数不参与权重衰减
        # BN是固定C,[B,H,W]进行归一化处理（处理为均值0,方差1的正太分布上），适用于CNN
        # LN是固定N,[C,H,W]进行归一化处理，适用于RNN（BN适用于固定深度的前向神经网络，而RNN因输入序列长度不一致而深度不固定，因此BN不合适，而LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作）
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

        # 配置优化器和warmup机制
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.train_steps//args.gradient_accumulation_steps)

        global_step = 0

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_acc=0
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        bar = tqdm(range(num_train_optimization_steps),total=num_train_optimization_steps)
        train_dataloader=cycle(train_dataloader) # 循环遍历

        # 先做一个eval
        for file in ['dev.csv']:
            inference_labels = []
            gold_labels = []
            inference_logits = []
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training=True)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, args.split_num,
                                                         False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            # Run prediction for full data 准备验证集的dataloader
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
            # 开启预测模式（不用dropout和BN）
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                # 将数据放在GPU上
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                # 禁止进行梯度更新
                with torch.no_grad():
                    tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                  attention_mask=input_mask,
                                                  labels=label_ids)
                    # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(np.argmax(logits, axis=1))
                gold_labels.append(label_ids)
                inference_logits.append(logits)
                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            gold_labels = np.concatenate(gold_labels, 0)
            inference_logits = np.concatenate(inference_logits, 0)
            model.train()
            eval_loss = eval_loss / nb_eval_steps # 计算验证集的预测损失
            eval_accuracy = accuracy(inference_logits, gold_labels) # 计算验证集的预测准确性

            result = {'eval_loss': eval_loss,
                      'eval_F1': eval_accuracy,
                      'global_step': global_step
                      }
            # 将验证集的预测评价写入到evel_results.txt中
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as writer:
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write('*' * 80)
                writer.write('\n')
            # 如果当前训练的模型表现最佳，则保存该模型
            if eval_accuracy > best_acc and 'dev' in file:
                print("=" * 80)
                print("Best F1", eval_accuracy)
                print("Saving Model......")
                best_acc = eval_accuracy
                # Save a trained model
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
                print("=" * 80)
            else:
                print("=" * 80)

        model.train()

        # 分batch循环迭代训练模型
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            loss, _ = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
            nb_tr_examples += input_ids.size(0)
            del input_ids, input_mask, segment_ids, label_ids
            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.fp16 and args.loss_scale != 1.0:
                loss = loss * args.loss_scale
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            train_loss=round(tr_loss*args.gradient_accumulation_steps/(nb_tr_steps+1),4)
            bar.set_description("loss {}".format(train_loss))

            nb_tr_steps += 1

            # 用FP16去做反向传播
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            # 梯度累计后进行更新
            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    # modify learning rate with special warm up BERT uses
                    # if args.fp16 is False, BertAdam is used that handles this automatically
                    lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                optimizer.step() # 梯度更新
                scheduler.step() # 梯度更新
                optimizer.zero_grad() # 清空现有梯度，避免累计
                global_step += 1

            # 每隔args.eval_steps*args.gradient_accumulation_steps，打印训练过程中的结果
            if (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0
                logger.info("***** Report result *****")
                logger.info("  %s = %s", 'global_step', str(global_step))
                logger.info("  %s = %s", 'train loss', str(train_loss))

            # 每隔args.eval_steps*args.gradient_accumulation_steps，预测验证集并评估结果
            if args.do_eval and step>num_train_optimization_steps*args.not_do_eval_steps and (step + 1) %(args.eval_steps*args.gradient_accumulation_steps)==0:
                for file in ['dev.csv']:
                    inference_labels=[]
                    gold_labels=[]
                    inference_logits=[]
                    eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = True)
                    eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,args.split_num,False)
                    all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
                    all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
                    all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
                    all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)


                    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)

                    logger.info("***** Running evaluation *****")
                    logger.info("  Num examples = %d", len(eval_examples))
                    logger.info("  Batch size = %d", args.eval_batch_size)

                    # Run prediction for full data
                    eval_sampler = SequentialSampler(eval_data)
                    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                    model.eval()
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                        input_ids = input_ids.to(device)
                        input_mask = input_mask.to(device)
                        segment_ids = segment_ids.to(device)
                        label_ids = label_ids.to(device)


                        with torch.no_grad():
                            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, labels=label_ids)
                            # logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                        logits = logits.detach().cpu().numpy()
                        label_ids = label_ids.to('cpu').numpy()
                        inference_labels.append(np.argmax(logits, axis=1))
                        gold_labels.append(label_ids)
                        inference_logits.append(logits)
                        eval_loss += tmp_eval_loss.mean().item()
                        nb_eval_examples += input_ids.size(0)
                        nb_eval_steps += 1

                    gold_labels=np.concatenate(gold_labels,0)
                    inference_logits=np.concatenate(inference_logits,0)
                    model.train()
                    eval_loss = eval_loss / nb_eval_steps
                    eval_accuracy = accuracy(inference_logits, gold_labels)

                    result = {'eval_loss': eval_loss,
                              'eval_F1': eval_accuracy,
                              'global_step': global_step,
                              'loss': train_loss}

                    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                    with open(output_eval_file, "a") as writer:
                        for key in sorted(result.keys()):
                            logger.info("  %s = %s", key, str(result[key]))
                            writer.write("%s = %s\n" % (key, str(result[key])))
                        writer.write('*'*80)
                        writer.write('\n')
                    if eval_accuracy>best_acc and 'dev' in file:
                        print("="*80)
                        print("Best F1",eval_accuracy)
                        print("Saving Model......")
                        best_acc=eval_accuracy
                        # Save a trained model
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print("="*80)
                    else:
                        print("="*80)

    # 预测测试集
    if args.do_test:
        del model
        gc.collect() # 清理内存
        args.do_train=False # 停止训练
        # 载入训练好的的最佳模型文件
        model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"),args,config=config)
        if args.fp16:
            # nn.Module中的half()方法将模型中的float32转化为float16
            model.half()
        model.to(device) # 将模型放在GPU上

        # 设置GPU训练方式
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        #  预测验证集和测试集
        for file,flag in [('dev.csv','dev'),('SP_test.csv','SP_test'),('SC_test.csv','SC_test')]:
            inference_labels=[]
            gold_labels=[]
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training = False)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length,args.split_num,False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)


            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(logits)
                gold_labels.append(label_ids)
            gold_labels=np.concatenate(gold_labels,0)
            logits=np.concatenate(inference_labels,0)
            print(flag, accuracy(logits, gold_labels))
            # 保存预测结果文件
            if flag=='SP_test':
                df=pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0']=logits[:,0]
                df['label_1']=logits[:,1]
                df[['qid','label_0','label_1']].to_csv(os.path.join(args.output_dir, "sub_SP.csv"),index=False)
            if flag=='SC_test':
                df=pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0']=logits[:,0]
                df['label_1']=logits[:,1]
                df[['qid','label_0','label_1']].to_csv(os.path.join(args.output_dir, "sub_SC.csv"),index=False)
            if flag == 'dev':
                df = pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0'] = logits[:, 0]
                df['label_1'] = logits[:, 1]
                df[['label_0', 'label_1']].to_csv(os.path.join(args.output_dir, "sub_dev.csv"),index=False)
    # 只预测验证集
    if args.predict_eval:
        del model
        gc.collect()
        args.do_train = False
        model = BertForSequenceClassification.from_pretrained(os.path.join(args.output_dir, "pytorch_model.bin"), args,
                                                              config=config)
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        for file, flag in [('dev.csv', 'dev')]:
            inference_labels = []
            gold_labels = []
            eval_examples = read_examples(os.path.join(args.data_dir, file), is_training=False)
            eval_features = convert_examples_to_features(eval_examples, tokenizer, args.max_seq_length, args.split_num,
                                                         False)
            all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
            all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
            all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
            all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                   attention_mask=input_mask).detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                inference_labels.append(logits)
                gold_labels.append(label_ids)
            gold_labels = np.concatenate(gold_labels, 0)
            logits = np.concatenate(inference_labels, 0)
            print(flag, accuracy(logits, gold_labels))
            if flag == 'dev':
                df = pd.read_csv(os.path.join(args.data_dir, file))
                df['label_0'] = logits[:, 0]
                df['label_1'] = logits[:, 1]
                df[['label_0', 'label_1']].to_csv(os.path.join(args.output_dir, "sub_dev.csv"),
                                                                   index=False)

if __name__ == "__main__":
    main()
