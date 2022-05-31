from language_chatbot.preprocess import get_preproc_features
from language_chatbot.data import getting_yaml_data
from gensim.models import Word2Vec
import gensim.downloader
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd


def embed_sentence(word2vec, sentence):
    '''
    returns a matrix that corresponds to the embedding of the full sentence
    '''
    embedded_sentence = []
    for word in sentence:
        if word in word2vec.wv:
            embedded_sentence.append(word2vec.wv[word])

    return np.array(embedded_sentence)


def embedding(word2vec, sentences):
    '''
    returns a list of embedded sentences (each sentence is a matrix)
    '''
    embed = []

    for sentence in sentences:
        embedded_sentence = embed_sentence(word2vec, sentence)
        embed.append(embedded_sentence)

    return embed


def word2vec_fun(data=getting_yaml_data(), value=0, padding='pre', vector_size=100, window=5, min_count=1, workers=3):
    '''
    Transforms the proprocessed data to word2vec
    the output included: (1) padding, (2) word2vec, (3) pretrained embedding
    '''

    # importing data and preprocessing data
    prep_X = get_preproc_features(data['patterns'])
    prep_X_list = prep_X.values.tolist()

    # load a pretrained embedding
    model_wiki = gensim.downloader.load('glove-wiki-gigaword-50')

    # This line trains an entire embedding for the words in your train set
    word2vec = Word2Vec(sentences=prep_X_list, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

    embed = pd.Series(embedding(word2vec, prep_X_list))

    # Padding
    X_pad = pad_sequences(embed, dtype='float32', padding=padding, value=value)

    return X_pad, word2vec, model_wiki


if __name__ == "__main__":
    print(word2vec_fun())
