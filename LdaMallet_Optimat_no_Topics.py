"""
LDA MALLET model using gensim
Finding the Optimal Number of Topics
Using Coherence Performance Measure
"""
### WebSite that I followed
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
#https://stackoverflow.com/questions/55789477/how-to-predict-test-data-on-gensim-topic-modelling
import json
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
# optional
# import loggingEnable logging for gensim -
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

with open('./data/all_data.json') as all_data:
    all_data=json.load(all_data)


print(f"data length= {len(all_data)}") #1219

# create a dictionary for data
dictionary = corpora.Dictionary(all_data)


# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in all_data]



###################################################
##              Building LDA Mallet Model
###################################################

mallet_path='/home/saba/PycharmProjects/testing/venv/lib/python3.6/site-packages/mallet-2.0.8/bin/mallet'




###################################################
##        How to find the optimal number of topics for LDA
###################################################
# finding the optimal number of topics is to build many LDA models
# with different values of number of topics (k)
# and pick the one that gives the highest coherence value.

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Can take a long time to run.
# start=2; limit=40; step=6
start=2; limit=30; step=1
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=doc_term_matrix, texts=all_data, start=start, limit=limit, step=step)

# Show graph
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


###########  OutPut
# Num Topics = 2  has Coherence Value of 0.3689
# Num Topics = 3  has Coherence Value of 0.4302
# Num Topics = 4  has Coherence Value of 0.4679
# Num Topics = 5  has Coherence Value of 0.4588
# Num Topics = 6  has Coherence Value of 0.4756
# Num Topics = 7  has Coherence Value of 0.4731
# Num Topics = 8  has Coherence Value of 0.4729
# Num Topics = 9  has Coherence Value of 0.4821
# Num Topics = 10  has Coherence Value of 0.4918
# Num Topics = 11  has Coherence Value of 0.4843
# Num Topics = 12  has Coherence Value of 0.4732
# Num Topics = 13  has Coherence Value of 0.4766
# Num Topics = 14  has Coherence Value of 0.4856
# Num Topics = 15  has Coherence Value of 0.4887
# Num Topics = 16  has Coherence Value of 0.4808
# Num Topics = 17  has Coherence Value of 0.4865
# Num Topics = 18  has Coherence Value of 0.4657
# Num Topics = 19  has Coherence Value of 0.505
# Num Topics = 20  has Coherence Value of 0.497
# Num Topics = 21  has Coherence Value of 0.5044
# Num Topics = 22  has Coherence Value of 0.4955
# Num Topics = 23  has Coherence Value of 0.489
# Num Topics = 24  has Coherence Value of 0.4968
# Num Topics = 25  has Coherence Value of 0.4954
# Num Topics = 26  has Coherence Value of 0.4969
# Num Topics = 27  has Coherence Value of 0.5002
# Num Topics = 28  has Coherence Value of 0.4806
# Num Topics = 29  has Coherence Value of 0.4854







