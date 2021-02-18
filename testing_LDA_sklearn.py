"""
Use Sklearn package to build the LDA model
Then use Grid Search to find the best topic we have

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
    for topic_idx, topic in enumerate(model.components_):
        print ("Topic %d:" % (topic_idx))
        print (" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

#
# with open('./data/train_data.json') as training:
#     train=json.load(training)
#
# with open('./data/test_data.json') as testing:
#     test=json.load(testing)


########### ver2, all text
# with open('./data/train_data_ver2.json') as training:
#     train=json.load(training)
#
# with open('./data/test_data_ver2.json') as testing:
#     test=json.load(testing)

# using filter(), to remove None values in list
# train= list(filter(None, train))
# test= list(filter(None, test))

# print(f"tain data length after removing None= {len(train)}") #1217
# print(f"test data length after removing None= {len(test)}")  #404

#to use LDA in Sklearn, I need to process the data
train_detokenize=list()

with open('./data/all_data.json') as testing:
    all_data=json.load(testing)

#detoken each sublist (each document)
for doc_token in all_data:
    doc_detoken=TreebankWordDetokenizer().detokenize(doc_token)
    train_detokenize.append(doc_detoken)


#to use LDA in Sklearn, I need to process the data
test_detokenize=list()

#detoken each sublist (each document)
for doc_token in all_data:
    doc_detoken=TreebankWordDetokenizer().detokenize(doc_token)
    test_detokenize.append(doc_detoken)


#Create the Document-Word matrix
vectorizer = CountVectorizer(analyzer='word',
                            # min_df=10,# minimum reqd occurences of a word
                              )
data_vectorized = vectorizer.fit_transform(train_detokenize)
tf_feature_names = vectorizer.get_feature_names()

# #Check the Sparsicity
# # Materialize the sparse data
# data_dense = data_vectorized.todense()
#
# # Compute Sparsicity = Percentage of Non-Zero cells
# #So 1 percentage, we have sparse matrix, I think it is good
# print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

# no_topics=5
# no_top_words = 15
# # Build LDA Model using sklearn
# lda_model = LatentDirichletAllocation(n_components=no_topics,               # Number of topics
#                                       max_iter=10,# Max learning iterations
#                                       #learning_method='online',
#                                       #random_state=100,# Random state
#                                       #batch_size=500,# n docs in each learning iter
#                                       evaluate_every = -1,# compute perplexity every n iters, default: Don't
#                                      # n_jobs = -1,# Use all available CPUs
#                                      )
#
# lda_output = lda_model.fit_transform(data_vectorized)
# # print(lda_model)  # Model attributes
#
# # Log Likelyhood: Higher the better
# #Diagnose model performance with perplexity and log-likelihood
# print("Log Likelihood: ", lda_model.score(data_vectorized))# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
# print("Perplexity: ", lda_model.perplexity(data_vectorized))# See model parameters

#Use GridSearch to determine the best LDA model.
# Define Search Param
search_params = {'n_components': [4,5,6,8,9,10,11,12,13,14,16,18, 20,22,24,25]}


# Init the Model
lda = LatentDirichletAllocation()
model=GridSearchCV(lda, param_grid=search_params)


# Do the Grid Search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_# Model Parameters
print("Best Model's Params: ", model.best_params_)# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# Get Log Likelyhoods from Grid Search Output
#n_topics = [5,6,8,10,12,14,16,18, 20,22,24,25]

# print(model.cv_results_.keys())
# print(model.cv_results_)


#log_likelyhoods_6 = [round(gscore.mean_validation_score) for gscore in model.cv_results_ if gscore.parameters['n_components']==6]
# log_likelyhoods_8 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==8]
# log_likelyhoods_10 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==10]
# log_likelyhoods_12 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==12]
# log_likelyhoods_14 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==14]
# log_likelyhoods_16 = [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==16]
# log_likelyhoods_18= [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==18]
# log_likelyhoods_20= [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==20]
# log_likelyhoods_22= [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==22]
# log_likelyhoods_24= [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==24]
# log_likelyhoods_25= [round(gscore.mean_validation_score) for gscore in model.grid_scores_ if gscore.parameters['n_components']==25]


# Show graph
# plt.figure(figsize=(12, 8))
# plt.plot(n_topics, log_likelyhoods_6, label='6')
# plt.plot(n_topics, log_likelyhoods_8, label='8')
# plt.plot(n_topics, log_likelyhoods_10, label='10')
# plt.plot(n_topics, log_likelyhoods_12, label='12')
# plt.plot(n_topics, log_likelyhoods_14, label='14')
# plt.plot(n_topics, log_likelyhoods_16, label='16')
# plt.plot(n_topics, log_likelyhoods_18, label='18')
# plt.plot(n_topics, log_likelyhoods_20, label='20')
# plt.plot(n_topics, log_likelyhoods_22, label='22')
# plt.plot(n_topics, log_likelyhoods_24, label='24')
# plt.plot(n_topics, log_likelyhoods_25, label='25')
# plt.title("Choosing Optimal LDA Model")
# plt.xlabel("Num Topics")
# plt.ylabel("Log Likelyhood Scores")
# plt.legend(title='number of topics', loc='best')
# plt.show()


pdb.set_trace()
# pprint(lda_model.get_params())
#
# # Create Document - Topic Matrix
# lda_output = lda_model.transform(data_vectorized)
# display_topics(lda_model, tf_feature_names, no_top_words)

#




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
#
#
#
#
#
#
#
# # from sklearn.decomposition import LatentDirichletAllocation
# # lda_model = LatentDirichletAllocation(n_topics=5,               # Number of topics
# #                                       max_iter=10,               # Max learning iterations
# #                                       learning_method='online',
# #                                       # random_state=100,          # Random state
# #                                       batch_size=128,            # n docs in each learning iter
# #                                       # evaluate_every = -1,       # compute perplexity every n iters, default: Don't
# #                                       # n_jobs = -1,               # Use all available CPUs
# #                                      )
# #
# # lda_output = lda_model.fit_transform(doc_term_matrix)
# #print(lda_model)  # Model attributes



# """
# search_params = {'n_components': [6,8,10,12,14,16,18, 20,22,24,25]}
# """
# tain data length after removing None= 1219
# test data length after removing None= 407
# Best Model's Params:  {'n_components': 6}
# Best Log Likelihood Score:  -940439.4508637034
# Model Perplexity:  2647.4423784184573
