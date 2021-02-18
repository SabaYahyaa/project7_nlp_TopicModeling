"""

I will create the X numerical features for the all_data
I use   TFIDF, and I will Store the model to deal for a new coming data
"""
import numpy as np
import json
import pdb
import gensim
from gensim import corpora
from gensim.models import CoherenceModel #to compute the performace measure for lda
from pprint import pprint
import pandas as pd
#ignore warning
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import joblib



def tfidf_vec(all_data):
    """
    :param all_data: is the prepocessed text, it is a list
                                Each element in a list is a list of tokens that is preprocessed before
                                Note: we can use Tfift to process also the text
    :return: X numerical features, and also the saved model
            The saved model can be used later to transform the text preprocessed data to numerical features
    """

    #to use LDA in Sklearn, I need to process the data
    data_detokenize=list()

    # tfid need preprocessed text, not token, for that I detoken the data that I have
    # note we can do preprocess using tfid
    #detoken each sublist (each document)
    for doc_token in all_data:
        doc_detoken=TreebankWordDetokenizer().detokenize(doc_token)
        # print(doc_detoken)
        # print(type(doc_detoken))
        # pdb.set_trace()
        data_detokenize.append(doc_detoken)
    # text=pd.DataFrame({'text':data_detokenize})
    # text=list(text.values)
    # print(type(data_detokenize))
    # print(type(data_detokenize[0]))
    #Note strip_accents = "ascii" to get only the englis words
    tf_vect = TfidfVectorizer(analyzer='word',stop_words= 'english', strip_accents = "ascii")
    X = tf_vect.fit_transform(data_detokenize)

    features_names=tf_vect.get_feature_names()
    # print(len(features_names))
    X=X.toarray()

    #print(features_names)
    #save the model, to use it later for new coming text
    joblib.dump(tf_vect, '{}.pkl'.format('./data/tf_vect'))
    return (X)

if __name__=='__main__':
    """
    Note: We create numerical features for all DATA that we have
    Note: all_data.json is the news_data.json that is preprocessed and saved
    We can APPLY this function on preprocessed data, that is stored is a list
    Each element in the list should be a list of tokens
    """

    #get the stored preprocessed text that is created by create_AllData_train_test.py
    with open('./data/all_data.json') as all_data:
        all_data=json.load(all_data)


    print(f"data length= {len(all_data)}") #1219
    X_numerical_features=tfidf_vec(all_data)
    print(X_numerical_features.shape) #(1626, 24959)


