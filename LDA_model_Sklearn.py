"""
Use Sklearn package to build the LDA model
Then use Grid Search to find the best topic we have
Note////// I used Gensium, it is better

"""
#Note:   tfidf with non negative matric factorization.  (cleaning)


import json
import numpy as np
import pandas as pd
import re, nltk, spacy, gensim# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint# Plotting tools
# import pyLDAvis
# import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import pdb
from nltk.tokenize.treebank import TreebankWordDetokenizer


def display_topics(model, feature_names, no_top_words):
    #note: model.components_   ,finds all the words for each topic
    #      model.component_[0] ,finds all the words for topic 1
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))




#to use LDA in Sklearn, I need to process the data
data_detokenize=list()

with open('./data/all_data.json') as all_data:
    all_data=json.load(all_data)



#detoken each sublist (each document)
for doc_token in all_data:
    doc_detoken=TreebankWordDetokenizer().detokenize(doc_token)
    data_detokenize.append(doc_detoken)


#Create the Document-Word matrix
# count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
vectorizer = CountVectorizer(analyzer='word',
                             stop_words='english'
                            # min_df=10,# minimum reqd occurences of a word
                              )
data_vectorized = vectorizer.fit_transform(data_detokenize)
#get feature name pass an ID to each word that we want to fetch
tf_feature_names = vectorizer.get_feature_names()

#Check the Sparsicity
# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
#So 1 percentage, we have sparse matrix, I think it is good
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

no_topics=2; no_top_words = 15
# Build LDA Model using sklearn
lda_model = LatentDirichletAllocation(n_components=no_topics,               # Number of topics
                                      max_iter=10,# Max learning iterations
                                      #learning_method='online',
                                      #random_state=100,# Random state
                                      #batch_size=500,# n docs in each learning iter
                                      evaluate_every = -1,# compute perplexity every n iters, default: Don't
                                     # n_jobs = -1,# Use all available CPUs
                                     )

lda_output = lda_model.fit_transform(data_vectorized)
# print(lda_model)  # Model attributes
#
# # Log Likelyhood: Higher the better
# #Diagnose model performance with perplexity and log-likelihood
# print("Log Likelihood: ", lda_model.score(data_vectorized))# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
# print("Perplexity: ", lda_model.perplexity(data_vectorized))# See model parameters


#get the words of the first topic
first_topic = lda_model.components_[0]
print(f"first topic words is {first_topic}")

top_topic_words = first_topic.argsort()[-15:]




#print(display_topics(lda_model, tf_feature_names, no_top_words))#model, feature_names, no_top_words
print('-------------------------------------')
print("     the words in each topic         ")
print('-------------------------------------')
for i,topic in enumerate(lda_model.components_):
    print(f'Top 10 words for topic #{i}:')
    print([tf_feature_names[i] for i in topic.argsort()[-10:]])
    print('\n')

print('-------------------------------------')
print("     store the topic for the text    ")
print('-------------------------------------')
# use LDA.transform() and pass document term matrix, to assign prob of topic to each document
topic_values = lda_model.transform(data_vectorized)
print(topic_values.shape)



















# pprint(lda_model.get_params())
#
# # Create Document - Topic Matrix
# lda_output = lda_model.transform(data_vectorized)
# display_topics(lda_model, tf_feature_names, no_top_words)






# # column names
# topicnames = ["Topic" + str(i) for i in range(lda_model.n_topics)]
#
# # index names
# docnames = ["Doc" + str(i) for i in range(len(train_detokenize))]
#
# # Make the pandas dataframe
# df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
#
# # Get dominant topic for each document
# dominant_topic = np.argmax(df_document_topic.values, axis=1)
# df_document_topic['dominant_topic'] = dominant_topic
#
# # Styling
# def color_green(val):
#     color = 'green' if val > .1 else 'black'
#     return 'color: {col}'.format(col=color)
#
# def make_bold(val):
#     weight = 700 if val > .1 else 400
#     return 'font-weight: {weight}'.format(weight=weight)
#
# # Apply Style
# df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
# print(df_document_topics)




