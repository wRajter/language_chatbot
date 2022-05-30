from language_chatbot.preprocess import get_preproc_features
from language_chatbot.data import getting_yaml_data
from gensim.models import Word2Vec
import gensim.downloader
from tensorflow.keras.preprocessing.sequence import pad_sequences




def word2vec_fun(data=getting_yaml_data()):
    '''transforming the proprocessed data to word2vec'''

    # importing data and preprocessing data
    prep_data = get_preproc_features(data[0:1])

    # This line trains an entire embedding for the words in your train set
    word2vec = Word2Vec(sentences=prep_data, vector_size=10)

    # load a pretrained embedding
    model_wiki = gensim.downloader.load('glove-wiki-gigaword-50')

    # Padding
    X_pad = pad_sequences(word2vec, dtype='float32', padding='post', value=-1000)


    return word2vec, model_wiki, X_pad
