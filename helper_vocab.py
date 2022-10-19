# AIM: To create a helper file for vocabulary processing/creation.

import string
import re
from os import listdir
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


directory_pos = 'Data/Raw/pos' # file address for positive sentences
directory_neg = 'Data/Raw/neg' # file address for positive sentences


'''
Note: Last 100 negative reviews as well as last 100 positive reviews will be used for model testing.
'''


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


def add_doc_to_vocab(path, vocab_counter):
    '''
    Function to load the document, clean the document to convert into tokens, and then add the tokens to vocab.
    
    Parameters:
        path (string): Path of the document where it is stored.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        
    Returns:
        tokens (list): List of all the tokens.
    '''
    doc = load_doc(path)
    tokens = clean_doc(doc)
    vocab_counter.update(tokens)
    


def process_docs(directory, vocab_counter):
    '''
    Function to pass the selected training files to add_doc_to_vocab().
    
    Parameters:
        directory (string): Directory of the document where it is stored.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        
    Returns:
        None.
    '''
    for filename in listdir(directory): # walk through all the files in the folder
        if filename.startswith('cv9'): # skip any reviews in the test set
            continue
        path = directory + '/' + filename
        add_doc_to_vocab(path, vocab_counter)


def tokens_min_occurance(min_occurance, vocab_counter, vocab_filename):
    '''
    Function to keep tokens having a minimum occurance.
    
    Parameters:
        min_occurance (int): Minimum occurance of a particular token.
        vocab_counter (dict): Dictionary of tokens and their occurances.
        vocab_filename (string): Name of the vocab file to store tokens.
        
    Returns:
        tokens (list): List of tokens with minimum occurance.
    '''
    tokens = [k for k,c in vocab_counter.items() if c>=min_occurance]
    save_token_list(tokens, vocab_filename)
    return tokens



def save_token_list(tokens, vocab_filename):
    '''
    Function to save token list to file.
    
    Parameters:
        tokens (list): List of tokens.
        vocab_filename (string): Name of the vocab file to store tokens.
        
    Returns:
        None.
    '''
    data = '\n'.join(tokens) # covert lines to a single blob of text
    file = open(vocab_filename, 'w')
    file.write(data)
    file.close()

    
if __name__ == '__main__':  
    min_occurance = 2
    vocab_counter = Counter() # define vocab counter
    process_docs(directory_pos, vocab_counter) # add all positive label documents to vocab
    process_docs(directory_neg, vocab_counter) # add all negative label documents to vocab
    print('Vocabulary Length (all tokens present in the document): ', len(vocab_counter))
    save_token_list(vocab_counter, vocab_filename='Data/Vocab/vocab_all_occ.txt') #saving list of all the tokens in a file
    tokens_min_occurance = tokens_min_occurance(min_occurance, vocab_counter, vocab_filename='Data/Vocab/vocab_min_occ.txt')
    print('Vocabulary Length (tokens with minimum occurance of ', min_occurance,'): ', len(tokens_min_occurance))

    
'''
Note: Use the tokens_min_occurance() function to explore the vocabulary (say, smaller vocabulary or larger vocabulary) to achieve better performance.
'''
