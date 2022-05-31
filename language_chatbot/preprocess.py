from data import getting_yaml_data

import string
import pandas as pd

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string


data = getting_yaml_data()
#data = pd.read_csv('language_chatbot/data/conver_df.csv')   #Alternative way of loading the data from the data (ABSSOLUTE PATH)
X = data['patterns']
y = data[['tag']]
columns = data['tag'].unique()

def lower(X):
    '''Function that returns panda series of lower case strings'''
    X = X.str.lower()
    return X

def remove_punctuations(text):
    '''Function that returns panda series with punctuation removed'''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def remove_numbers(X):
    '''Function that returns a panda series with numbers removed'''
    X = X.apply(lambda x: ''.join(word for word in x if not word.isdigit()))
    return X

def remove_stop_words(X):
    '''Function that returns a panda series of a list of words with stop words removed'''
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(X)
    X = [w for w in word_tokens if not w in stop_words]
    return X

def lemmatizing(text):
    '''Function that returns a panda series of STRINGS of lemmatized words'''
    lemmatizer = WordNetLemmatizer()
    new_text = [lemmatizer.lemmatize(word) for word in text]
    return new_text

def get_preproc_features(X):
    '''All functions above combined to return a full preprocessed panda series :-)'''
    X = lower(X)
    X = X.apply(remove_punctuations)
    X = remove_numbers(X)
    for index, value in X.iteritems():
        X[index] = remove_stop_words(value)
    for index, value in X.iteritems():
        X[index] = lemmatizing(value)

    return X

def get_preproc_target(df = data):
    return pd.get_dummies(df.tag)


if __name__ == '__main__':
    output = get_preproc_target()
    print(output)
