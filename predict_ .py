# AIM: To create a pipeline for predicting the reviews on final trained model.

import string
import re
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.models import load_model


vocab_filename = 'Data/Vocab/vocab_all_occ.txt' # vocab filename with its location


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


def process_docs(directory, vocab_counter):
    '''
    Function to load all documents in a directory and return list of tokens.
    
    Parameters:
        directory (string): Directory of the document where it is stored.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        
    Returns:
        lines (list): List of tokens.
    '''
    lines = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        line = doc_to_line(path, vocab_counter)
        lines.append(line)
    return lines


def load_clean_dataset(vocab_counter):
    '''
    Function to load and clean a dataset.
    
    Parameters:
        vocab_counter (dict): Dictionary of tokens and their occurances.
        
    Returns:
        docs (list): List of tokens.
        labels (list): List of labels.
    '''
    pos = process_docs('Data/Raw/pos', vocab_counter)
    neg = process_docs('Data/Raw/neg', vocab_counter)   
    docs = neg + pos
    labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))] # prepare labels
    return docs, labels


def predict_sentiment(review_text, vocab, tokenizer, model):
    '''
    Function to classify a document as positive or negative.
    
    Parameters:
        review_text (string): Review to classify as positive or negative.
        vocab (string): Vocabulary.
        tokenizer (): 
        model (h5): Trained final model for prediction.
        
    Returns:
        percent_pos (int): Percentage with which the model predicts positive review.
        label(NEGATIVE/POSITIVE) (string): Predicted positive or negative review.
    '''
    tokens = clean_doc(review_text) # clean text
    tokens = [w for w in tokens if w in vocab] # filter by vocab
    line = ' '.join(tokens) # convert to line
    encoded = tokenizer.texts_to_matrix([line], mode='binary') # encode
    yhat = model.predict(encoded, verbose=0) # predict sentiment
    percent_pos = yhat[0, 0] 
    if round(percent_pos) == 0:
        return (1-percent_pos), 'NEGATIVE'
    return percent_pos, 'POSITIVE'


if __name__ == '__main__': 
    vocab = load_doc(vocab_filename) # load vocabulary
    vocab = set(vocab.split())
    train_docs, ytrain = load_clean_dataset(vocab)
    test_docs, ytest = load_clean_dataset(vocab)
    tokenizer = Tokenizer() # create tokenizer
    tokenizer.fit_on_texts(train_docs)
    saved_model = load_model('Models/best_model.h5') # load the saved model
    # test positive review
    text = 'One of the best movies i have seen. Recommended.'
    percent, sentiment = predict_sentiment(text, vocab, tokenizer, saved_model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
    # test negative review
    text = 'Bad movie. Worst screenplay.'
    percent, sentiment = predict_sentiment(text, vocab, tokenizer, saved_model)
    print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))