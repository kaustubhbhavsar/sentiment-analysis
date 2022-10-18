# AIM: To create a helper file for analysing neural network.

import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense
from pandas import DataFrame
import numpy


directory_pos = 'Data/Raw/pos' # file address for positive sentences
directory_neg = 'Data/Raw/neg' # file address for positive sentences
vocab_filename = 'Data/Vocab/vocab_min_occ.txt' # vocab filename with its location


def load_doc(filename):
    '''
    Function to load document into memory.
    
    Parameters:
        filename (string): Name of the file to open.
        
    Returns:
        text (string): All the text contained in the file.
    '''
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def clean_doc(doc):
    '''
    Function to turn a document into clean tokens.
    
    Parameters:
        doc (string): Document to convert into tokens.
        
    Returns:
        tokens (list): List of all the tokens.
    '''
    tokens = doc.split() # split into tokens by whitespace
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) # prepare regex for char filtering
    tokens = [re_punc.sub('', w) for w in tokens] # remove punctuation from each word
    tokens = [word for word in tokens if word.isalpha()] # remove remaining tokens that are not alphabetic
    stop_words = set(stopwords.words('english')) 
    tokens = [w for w in tokens if not w in stop_words] # filter out stop words
    return tokens


def doc_to_line(path, vocab_counter):
    '''
    Function to load document, clean and return string of tokens present in vocab.
    
     Parameters:
        path (string): Path of the document where it is stored.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        
    Returns:
        String of tokens separated by space.
    '''
    doc = load_doc(path)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab_counter]
    return ' '.join(tokens)

def process_docs(directory, vocab_counter, is_train):
    '''
    Function to load all documents in a directory and return list of tokens.
    
    Parameters:
        directory (string): Directory of the document where it is stored.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        is_train (boolean): If the document is a training document or testing document.
        
    Returns:
        lines (list): List of tokens.
    '''
    lines = list()
    for filename in listdir(directory):
        if is_train and filename.startswith('cv9'):
            continue
        if not is_train and not filename.startswith('cv9'):
            continue
        path = directory + '/' + filename
        line = doc_to_line(path, vocab_counter)
        lines.append(line)
    return lines


def load_clean_dataset(vocab_counter, is_train):
    '''
    Function to load and clean a dataset.
    
    Parameters:
        vocab_counter (dict): Dictionary of tokens and their occurances.
        is_train (boolean): If the document is a training document or testing document.
        
    Returns:
        docs (list): List of tokens.
        labels (list): List of labels.
    '''
    pos = process_docs('Data/Raw/pos', vocab_counter, is_train)
    neg = process_docs('Data/Raw/neg', vocab_counter, is_train)   
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))] # prepare labels
    return docs, labels


def define_model(n_words):
    '''
    Function to define the model.
    
    Parameters:
        n_words (): .
        
    Returns:
        model ( ): Neural network model.
    '''
    model = Sequential()
    model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def evaluate_mode(Xtrain, ytrain, Xtest, ytest):
    '''
    Function to evaluate a neural network model.
    
    Parameters:
        Xtrain (string/docs): Training data.
        ytrain (list): Training labels.
        Xtest (string/docs): Testing data.
        ytest (list): Testing labels.
        
    Returns:
        scores (list): List of scores.
    '''
    n_words = Xtest.shape[1]
    model = define_model(n_words)
    model.fit(Xtrain, ytrain, epochs=10, verbose=0)
    _, accuracy = model.evaluate(Xtest, ytest, verbose=0)
    return accuracy


def prepare_data(train_docs, test_docs, mode):
    '''
    Function to prepare bag of words encoding of docs.
    
    Parameters:
        train_docs (): 
        test_docs ():
        mode ():
        
    Returns:
        Xtrain (): 
        Xtest ():
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    Xtest = tokenizer.texts_to_matrix(test_docs, mode=mode)
    return Xtrain, Xtest


if __name__ == '__main__': 
    vocab = load_doc(vocab_filename) # load the vocabulary
    vocab = vocab.split()
    train_docs, ytrain = load_clean_dataset(vocab, True)
    test_docs, ytest = load_clean_dataset(vocab, False)
    modes = ['binary', 'count', 'tfidf', 'freq'] # list of modes
    for mode in modes:
        Xtrain, Xtest = prepare_data(train_docs, test_docs, mode) # prepare data for mode
        accuracy = evaluate_mode(numpy.array(Xtrain), numpy.array(ytrain), numpy.array(Xtest), numpy.array(ytest)) # evaluate for mode
        print('Mode: ', mode, ' | Accuracy: ', accuracy)