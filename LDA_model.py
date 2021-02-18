"""
LDA MODEL using Gensim
NOTE: I compared the coherence measure of LDA and LDA-MALLET,
I select LDA-MALLET since the coherence measure is higher
"""
#useful links
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#https://stackoverflow.com/questions/55789477/how-to-predict-test-data-on-gensim-topic-modelling
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

with open('./data/all_data.json') as all_data:
    all_data=json.load(all_data)

print(f"data length= {len(all_data)}")

# create a dictionary for data
dictionary = corpora.Dictionary(all_data)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in all_data]

num_topics=2; passes=50  #passes== number of iterations
###################################################
###################################################
###################################################
##              Building LDA Model
###################################################
###################################################
###################################################
#Running LDA Model
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.

lda_model = Lda(doc_term_matrix, num_topics=num_topics,
                id2word = dictionary,
                random_state=500,
                passes=passes)
#pprint(lda_model.print_topics(num_words=30))

print("--------------- TOPICs Using LDA------------------------------")
for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=30):
    print(str(i)+": "+ topic)
    print()



print("----------------Perplexity and Coherence Score for LDA----------------------------")
print('\nPerplexity: ', lda_model.log_perplexity(doc_term_matrix))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=all_data, dictionary=dictionary, coherence='c_v')
coherence_lda= coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
