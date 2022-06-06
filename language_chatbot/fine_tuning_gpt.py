# dataset, split dataset
# make imports
# call the pretrain model (dialo-GPT)
# call the tokenizer and encoding
# convert tokenized data to the right format (tensorflow)
# import trainer
# start training with the pre-train model
# evaluate
# save


import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TFTrainer, TFTrainingArguments
import tensorflow as tf

# reading dataset
all_rick = pd.read_csv('/home/lubo/code/wRajter/language_chatbot/raw_data/RickAndMortyScripts.csv')

# modifing dataset
contexted = []
n = 7
for i in range(n, len(all_rick['line'])):
  row = []
  prev = i - 1 - n # we additionally subtract 1, so row will contain current response and 7 previous responses
  for j in range(i, prev, -1):
    row.append(all_rick['line'][j])
  contexted.append(row)
columns = ['response', 'context']
columns = columns + ['context/'+str(i) for i in range(n-1)]
df = pd.DataFrame.from_records(contexted, columns=columns)

# split data
train_df, val_df = train_test_split(df, test_size = 0.1)


# call the pretrain model (dialo-GPT)
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# call the tokenizer and encoding
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

# tokeninzing
train_list = train_df[['response', 'context']].values.tolist()
val_list = val_df[['response', 'context']].values.tolist()

tokenizer.pad_token = tokenizer.eos_token

train_encodings = tokenizer(train_list, truncation=True, padding=True, return_tensors='pt')
val_encodings = tokenizer(val_list, truncation=True, padding=True, return_tensors='pt')

# converting encodings to tensorflow

tf_train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings)
))

tf_val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings)
))


checkpoint = tf.train.Checkpoint(step=tf.Variable(1))


# import trainer
training_arguments = TFTrainingArguments(output_dir='test_trainer',
                                       num_train_epochs=2,
                                       evaluation_strategy='steps',
                                       eval_steps=500,
                                       per_device_train_batch_size=8,
                                       per_device_eval_batch_size=8,
                                       seed=0,
                                       load_best_model_at_end=True)


# defining trainer
trainer = TFTrainer(
    model=model,
    args=training_arguments,
    train_dataset=tf_train_dataset,
    eval_dataset=tf_val_dataset,
)


# training
trainer.train()


# Compile and fit
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=tf.metrics.SparseCategoricalAccuracy(),
# )

# model_fitting = model.fit(tf_train_dataset, validation_data=tf_val_dataset, epochs=3)

# eval_ = model_fitting.eval()



if __name__ == '__main__':
    print('done')
