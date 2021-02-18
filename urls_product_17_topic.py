"""
NOTE: I will Suppose that the HR will have only urls and needs to find the topic from them
I will create the X numerical features for the all_data
I use   TFIDF, and I will Store it in df
"""
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import json
import pdb
import gensim
from gensim import corpora
from gensim.models import CoherenceModel #to compute the performace measure for lda
from pprint import pprint
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import pandas as pd
#ignore warning
# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, cohen_kappa_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#all_data is all the preproceesed data
with open('./data/all_data.json', 'r') as processed_all_text:
    all_processed_text=json.load(processed_all_text)


print(f"data length= {len(all_processed_text)}") #1626

processed_text = pd.Series(all_processed_text, name='Processed_text')
df = processed_text.to_frame()

#extract urls from news_data.json and added to the processed_text

with open("./data/news_data.json", "r") as f:
    papers10 = json.load(f)
papers = pd.json_normalize(papers10["data"])
#print(papers.columns.tolist())

#extract the text, date, url and conconate them with the preprocessed text
df['text']=papers['text'].values
df['url']=papers['url'].values
df['date']=papers['date'].values

#we need also the label from the Topic Modeling
#get the y from LDA-Mallet model,
topics_all=pd.read_csv("./data/topics_all_data_17.csv")
#print(topics_all.columns.tolist())  #['Unnamed: 0', 'Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df['Dominant_Topic']=topics_all['Dominant_Topic'].values
df['Topic_Perc_Contrib']=topics_all['Topic_Perc_Contrib'].values
df['Keywords']=topics_all['Keywords'].values

#save the df in csv file,
df.to_csv('./data/news_labeled_data.csv')
topic=15
to_be_filter_topic = df['Dominant_Topic'].apply(lambda x: x == topic)
filtered_url=df.loc[to_be_filter_topic, 'url'].values
#self.sales_data.loc[to_be_deleted_filter, 'price'] = None
print(len(filtered_url))
print(df.columns.tolist())
print(df['Dominant_Topic'].value_counts())
print(type(filtered_url))
pprint(filtered_url)
#
# pprint(df['Keywords'])





