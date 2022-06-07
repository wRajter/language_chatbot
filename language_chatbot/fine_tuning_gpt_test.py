# imports
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
from torch.utils.data import Dataset
from collections import Counter
import json



############
### DATA ###
############

data = pd.read_csv('/home/lubo/code/wRajter/language_chatbot/raw_data/RickAndMortyScripts.csv')

# class RickDataset(Dataset):
contexted = []
n = 7
for i in range(n, len(data['line'])):
  row = []
  prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses
  for j in range(i, prev, -1):
    row.append(data['line'][j])
  contexted.append(row)
columns = ['response', 'context']
columns = columns + ['context/'+str(i) for i in range(n-1)]
df = pd.DataFrame.from_records(contexted, columns=columns)

# split data
train_df, val_df = train_test_split(df, test_size = 0.2)
# print(len(train_df)) # ---> 1518
# print(len(val_df)) # ---> 380


##############
## ENCODING ##
##############

def get_counter_and_lens(data, tokenizer):
    flatten = lambda l: [item for sublist in l for item in sublist]
    toks = [tokenizer.tokenize(x) for x in data]

    return list(map(len, toks)), Counter(flatten(toks)), Counter(' '.join(data).split())

model_name = 'microsoft/DialoGPT-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
lens, tok_cnt, word_cnt = get_counter_and_lens(val_df[df.columns].apply(lambda x: ' '.join(x.astype(str)), axis = 1), tokenizer)


# call the pre-train model and tokenizer (dialo-GPT)
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")


# tokenizer.pad_token = tokenizer.eos_token

def construct_conv(row, tokenizer, eos = True):
    '''
    converts the context and response data into a single conversation string
    inspired by from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    this string will be separated by a special token that tells our model when a person is finished speaking
    tokenizer.encode(x) -> tokenization of each word in a row
    tokenizer.eos_token_id -> creates a special token (50256) between individual speeches
    '''
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

class DatasetEncoding(Dataset):
    def __init__(self, tokenizer=tokenizer):

        self.block_size = 512 - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        self.examples = []
        for _, row in val_df.iterrows():
            conv = construct_conv(row, tokenizer)
            if len(conv) > self.block_size: continue
            self.examples.append(conv)
        with open("data/encodings", "w") as fp:
            json.dump(self.examples, fp)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)






# creating a torch dataset object (tensors)
# val_encodings = [torch.tensor(data[x], dtype=torch.long) for x in range(len(data))]

# converting encodings to tensorflow
# tf_val_dataset = tf.data.Dataset.from_tensor_slices(data)



# import trainer
# training_arguments = TrainingArguments(output_dir='test_trainer',
#                                        num_train_epochs=2,
#                                        evaluation_strategy='steps',
#                                        eval_steps=500,
#                                        per_device_train_batch_size=8,
#                                        per_device_eval_batch_size=8,
#                                        seed=0,
#                                        load_best_model_at_end=True,
#                                        max_steps = 10)


# # defining trainer
# trainer = Trainer(
#     model=model,
#     args=training_arguments,
#     train_dataset=encodings,
# )


# # # training
# trainer.train()

if __name__ == '__main__':
    # print(len(lens))
    # print(tok_cnt)
    # print(word_cnt)
    test = DatasetEncoding()
    print('DONE - without any errors :)')
