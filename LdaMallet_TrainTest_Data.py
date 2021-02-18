"""
I will create LdaMallet to see how it work (correctly or not)
"""
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

import numpy as np
np.random.seed(42)

###### write to json file
with open('./data/train_data.json') as train:
    train = json.load(train)

print(f"data length= {len(train)}") #1219




# create a dictionary for data
dictionary = corpora.Dictionary(train)


# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in train]
num_topics=17

###################################################
###################################################
###################################################
##              Building LDA Mallet Model
###################################################
###################################################
###################################################
mallet_path='/home/saba/PycharmProjects/testing/venv/lib/python3.6/site-packages/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
                                             corpus=doc_term_matrix,
                                             num_topics=num_topics, id2word=dictionary)




# ###################################################
# ##               Performance Measure
# ###################################################

print("----------------Perplexity and Coherence Score for LDA-Mallet----------------------------")
# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=train, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)




# ###################################################
# ##               Topics
# ###################################################

print("----------------Topics for LDA-Mallet ----------------------------")
# Show Topics
model_topics=ldamallet.show_topics(formatted=False,  num_topics=num_topics, num_words=40)
pprint(ldamallet.print_topics(num_words=40))


###################################################
###################################################
#   Finding the dominant topic for each document
###################################################
###################################################
#what the topic of a document is about
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


#df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet , corpus=doc_term_matrix, texts=train)

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#
#
#
# # Show
# print(df_dominant_topic.head(10))
# print("-----------------------------------------------")
#
# print(df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib']].head(25))
# print("-----------------------------------------------")
# print(ldamallet[doc_term_matrix[0]]) # corpus[0] means the first document.
# print(ldamallet[doc_term_matrix[1]]) # corpus[0] means the first document.
# print(ldamallet[doc_term_matrix[2]]) # corpus[0] means the first document.
# df_dominant_topic.to_csv('./data/topics_train_data.csv')


###################################################
###################################################
#   LDA-Mallet on test data, this is will be done visually by ourself
###################################################
###################################################
with open('./data/test_data.json') as test:
    test= json.load(test)
print(f"data length= {len(test)}") #407



# create a dictionary of individual words and filter the dictionary
dictionary_new = gensim.corpora.Dictionary(test)

# # define the bow_corpus
bow_corpus_new = [dictionary_new.doc2bow(doc) for doc in np.array(test)]
a = ldamallet[bow_corpus_new[:len(bow_corpus_new)]]
print(a[0])
print(test[0])




#df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet , corpus=doc_term_matrix, texts=train)

# # Format
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#
#
#
# # Show
# print(df_dominant_topic.head(10))
# print("-----------------------------------------------")
#
# print(df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib']].head(25))
# print("-----------------------------------------------")
# print(ldamallet[doc_term_matrix[0]]) # corpus[0] means the first document.
# print(ldamallet[doc_term_matrix[1]]) # corpus[0] means the first document.
# print(ldamallet[doc_term_matrix[2]]) # corpus[0] means the first document.


# df_dominant_topic.to_csv('./data/topics_train_data.csv')


# pdb.set_trace()






















