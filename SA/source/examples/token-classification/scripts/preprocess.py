import sys

from pytorch_transformers import AutoTokenizer


# dataset = sys.argv[1]
# model_name_or_path = sys.argv[2]
# max_len = int(sys.argv[3])

dataset = r'/media/alvinai/Documents/baidu_nlp/data/GermEval/train.txt.tmp'
model_name_or_path = 'bert-base-multilingual-cased'
max_len = 128

subword_len_counter = 0

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# max_len -= tokenizer.num_special_tokens_to_add()
max_len -= 2
with open(dataset, "rt") as f_p:
    for line in f_p:
        line = line.rstrip()
        if not line:
            print(line)
            subword_len_counter = 0
            continue
        token = line.split()[0]
        current_subwords_len = len(tokenizer.tokenize(token))
        # Token contains strange control characters like \x96 or \x95
        # Just filter out the complete line
        if current_subwords_len == 0:
            continue
        if (subword_len_counter + current_subwords_len) > max_len:
            print("")
            print(line)
            subword_len_counter = current_subwords_len
            continue
        subword_len_counter += current_subwords_len
        print(line)