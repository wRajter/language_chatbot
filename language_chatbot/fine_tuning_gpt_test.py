# imports
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM




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

# call the pretrain model and tokenizer (dialo-GPT)
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

##############
## encoding ##
##############

tokenizer.pad_token = tokenizer.eos_token


def construct_conv(row, tokenizer, eos = True):
    # from: https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x, padding=True, return_tensors='tf') for x in row]))
    conv = flatten(conv)
    return conv

data = []
for _, row in val_df.iterrows():
    conv = construct_conv(row, tokenizer)
    data.append(conv)

# creating a torch dataset object (tensors)
# val_encodings = [torch.tensor(data[x], dtype=torch.long) for x in range(len(data))]

# converting encodings to tensorflow
tf_val_dataset = tf.data.Dataset.from_tensor_slices(data)



# import trainer
training_arguments = TrainingArguments(output_dir='test_trainer',
                                       num_train_epochs=2,
                                       evaluation_strategy='steps',
                                       eval_steps=500,
                                       per_device_train_batch_size=8,
                                       per_device_eval_batch_size=8,
                                       seed=0,
                                       load_best_model_at_end=True)


# defining trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=val_encodings
)


# training
trainer.train()

if __name__ == '__main__':
    print(data[1])
    print(type(data))
    print(len(data))
    print('DONE - without any errors :)')
