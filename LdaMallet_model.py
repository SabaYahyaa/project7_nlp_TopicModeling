
# ################################################
# #  get the topics for a given text
# #################################################

# import pdb
# import gensim
# import json
# import numpy as np
# ##save model
# # model.save('ldamallet_test.gensim')

# #load a model
# lda_mallet = gensim.models.wrappers.LdaMallet.load('ldamallet_test.gensim')
# with open('./data/all_data.json') as all_data:
#     all_data=json.load(all_data)
# unseen_data=[all_data[0]]
# print(unseen_data)
# print(type(unseen_data))
# # create a dictionary of individual words and filter the dictionary
# dictionary_new = gensim.corpora.Dictionary(unseen_data)

# # define the bow_corpus
# bow_corpus_new = [dictionary_new.doc2bow(doc) for doc in np.array(unseen_data)]

# a = lda_mallet[bow_corpus_new[:len(bow_corpus_new)]]
# print(a)
# pdb.set_trace()


"""
    LDA MALLET model using gensim
    the Optimal Number of Topics is 10 by using LdaMallet_Optimal_no_Topics
"""
# https://www.youtube.com/watch?v=l-lK_jWf3rI    TFIDF - Bag of Words Techniqu
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
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
import os
import numpy as np
np.random.seed(42)


###################################################
###################################################
#    Finding the dominant topic in each sentence
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




def labels_by_LdaMallet(all_data, num_topics, num_words):
    """
        :param all_data: is the preprocessed data list,
                    each element is a preproced document, is a list of tokens
        :param num_topics: is the optimal number of topics, the optimal number should be found using LdaMallet_Optimal_no_Topics
        :return: y label, the dominated topics for each document
        The function produce and stores the following:
        1. store the ladmallet model, this is will be used later to create topics for unseen data
        2. Perplexity and Coherence Score for LDA-Mallet
        3. the num_words words for each topic
        4. create a scv that shows the text, keywords, dominated topcs, and distribution for each topic
    """

    # create a dictionary for data
    dictionary = corpora.Dictionary(all_data)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in all_data]

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
                                                 num_topics=num_topics,
                                                 id2word=dictionary)


    #create a path to store the gensium model under the data folder
    p=os.path.splitext('./data/ldamallet_all_data_')[0] + str(num_topics)
    p=os.path.splitext(p)[0] + '.gensim'  #e.g., './data/ldamallet_all_data_17.gensim'
    ldamallet.save(p)

    # ##  save the model using gensim
    # ldamallet.save('./data/ldamallet_all_data_17.gensim')

#
# ############################################################################
# #####           visualization has to be in jupeternote book
# #############################################################################
# #for visualization, we shoud convert ldamallet to lad
# model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
#
#
# # #convert lda-mallet to lda using the folowing function
# # from gensim.models.ldamodel import LdaModel
# # import numpy
# #
# # def ldaMalletConvertToldaGen(mallet_model):
# #     model_gensim = LdaModel(id2word=mallet_model.id2word, num_topics=mallet_model.num_topics, alpha=mallet_model.alpha, eta=0, iterations=1000, gamma_threshold=0.001, dtype=numpy.float32)
# #     model_gensim.state.sstats[...] = mallet_model.wordtopics
# #     model_gensim.sync_state()
# #     return model_gensim
# #
# # converted_model = ldaMalletConvertToldaGen(ldamallet)
# #


#
# #note:  it starts with 1, not zero
# import pyLDAvis.gensim
#
# vis_data = pyLDAvis.gensim.prepare(model , doc_term_matrix, dictionary, sort_topics=False)
# #pyLDAvis.show(vis_data),   error, display nothing appears
#
#
# # in notebook
# #pyLDAvis.enable_notebook()
# # vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# # vis

    # ###################################################
    # ##               Performance Measure
    # ###################################################

    print("----------------Perplexity and Coherence Score for LDA-Mallet----------------------------")
    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=all_data, dictionary=dictionary, coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    # ###################################################
    # ##               Topics
    # ###################################################

    print("----------------Topics for LDA-Mallet ----------------------------")
    # Show Topics
    model_topics=ldamallet.show_topics(formatted=False,  num_topics=num_topics, num_words=num_words)
    #pprint(model_topics)
    #or
    pprint(ldamallet.print_topics(num_words=num_words))

    #call the format_topics_sentences to generate df
    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamallet , corpus=doc_term_matrix, texts=all_data)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

    #store the df in the data foldel
    # create a path to store the gensium model under the data folder
    pp = os.path.splitext('./data/topics_all_data_')[0] + str(num_topics)
    pp = os.path.splitext(pp)[0] + '.csv'  # e.g., './data/ldamallet_all_data_17.gensim'
    ldamallet.save(pp)

    #df_dominant_topic.to_csv('./data/topics_all_data_17.csv')

    # # Show
    # print(df_dominant_topic.head(10))
    # print("-----------------------------------------------")
    #
    # print(df_dominant_topic[['Dominant_Topic', 'Topic_Perc_Contrib']])
    # print("-----------------------------------------------")
    # print(ldamallet[doc_term_matrix[0]]) # corpus[0] means the first document.
    #
    # print(ldamallet[doc_term_matrix[1]]) # corpus[0] means the first document.


    # ###################################################
    # ###################################################
    # #   Topic distribution across documents
    # ###################################################
    # ###################################################
    # # Number of Documents for Each Topic
    # # Number of Documents for Each Topic
    # topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    #
    # # Percentage of Documents for Each Topic
    # topic_contribution = round(topic_counts/topic_counts.sum(), 4)
    #
    # # Topic Number and Keywords
    # topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    #
    # # Concatenate Column wise
    # df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    #
    # # Change Column names
    # df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
    #
    # # Show
    # pprint(df_dominant_topics)
    return (df_dominant_topic['Dominant_Topic'].values)  #return y, each label is the dominated topics

if __name__=='__main__':
    """
    Note: We create numerical labels 
    here, I create labels for all DATA that we have
    Note: all_data.json is the news_data.json that is preprocessed and saved
    We can APPLY this function on preprocessed data, that is stored is a list
    Each element in the list should be a list of tokens
    """
    num_topics = 2 #no of topics
    num_words=2 # the top words for each topic
    #get the stored preprocessed text that is created by create_AllData_train_test.py
    with open('./data/all_data.json') as all_data:
        all_data=json.load(all_data)


    print(f"data length= {len(all_data)}") #1219
    #get the label for each document using lda-mallet
    y=labels_by_LdaMallet(all_data,num_topics,num_words)
    print(len(y)) #(1626, 24959)




















# ####################################################
# # ####################################################
# # #####                   17 topics
# # ####################################################
# # ####################################################
#
# ----------------Perplexity and Coherence Score for LDA-Mallet----------------------------
#
# Coherence Score:  0.4890902512718885
# ----------------Topics for LDA-Mallet ----------------------------
# [(0,
#   '0.026*"video" + 0.014*"facebook" + 0.014*"content" + 0.014*"news" + '
#   '0.011*"photo" + 0.011*"fake" + 0.011*"people" + 0.010*"post" + 0.009*"face" '
#   '+ 0.009*"social" + 0.009*"medium" + 0.009*"online" + 0.008*"user" + '
#   '0.008*"software" + 0.008*"create" + 0.007*"image" + 0.007*"text" + '
#   '0.007*"work" + 0.007*"real" + 0.006*"make" + 0.006*"platform" + '
#   '0.006*"identify" + 0.006*"twitter" + 0.006*"generate" + 0.006*"person" + '
#   '0.006*"tool" + 0.006*"show" + 0.006*"speech" + 0.005*"review" + 0.005*"ad" '
#   '+ 0.005*"youtube" + 0.005*"remove" + 0.005*"human" + 0.005*"article" + '
#   '0.005*"call" + 0.005*"deepfakes" + 0.004*"time" + 0.004*"report" + '
#   '0.004*"include" + 0.004*"read"'),
#  (1,
#   '0.037*"algorithm" + 0.022*"percent" + 0.018*"decision" + 0.018*"system" + '
#   '0.015*"people" + 0.013*"machine" + 0.012*"make" + 0.012*"problem" + '
#   '0.012*"test" + 0.011*"human" + 0.010*"result" + 0.010*"bias" + '
#   '0.009*"found" + 0.009*"study" + 0.008*"set" + 0.007*"data" + 0.007*"case" + '
#   '0.007*"black" + 0.006*"woman" + 0.006*"high" + 0.006*"give" + 0.006*"tool" '
#   '+ 0.005*"individual" + 0.005*"risk" + 0.005*"analysis" + 0.005*"rate" + '
#   '0.005*"identify" + 0.005*"show" + 0.005*"automate" + 0.005*"lack" + '
#   '0.005*"issue" + 0.004*"question" + 0.004*"bad" + 0.004*"good" + '
#   '0.004*"change" + 0.004*"accurate" + 0.004*"accuracy" + 0.004*"level" + '
#   '0.004*"gender" + 0.004*"group"'),
#  (2,
#   '0.014*"researcher" + 0.012*"material" + 0.011*"data" + 0.010*"system" + '
#   '0.009*"design" + 0.009*"science" + 0.008*"research" + 0.008*"process" + '
#   '0.007*"set" + 0.007*"structure" + 0.007*"complex" + 0.006*"professor" + '
#   '0.006*"work" + 0.006*"engineering" + 0.006*"point" + 0.006*"energy" + '
#   '0.006*"paper" + 0.006*"algorithm" + 0.005*"approach" + 0.005*"cell" + '
#   '0.005*"mit" + 0.005*"make" + 0.005*"small" + 0.005*"property" + '
#   '0.005*"produce" + 0.005*"find" + 0.005*"step" + 0.005*"light" + '
#   '0.005*"machine" + 0.005*"large" + 0.005*"number" + 0.004*"high" + '
#   '0.004*"problem" + 0.004*"graph" + 0.004*"call" + 0.004*"technique" + '
#   '0.004*"method" + 0.004*"molecule" + 0.004*"generate" + 0.004*"change"'),
#  (3,
#   '0.020*"user" + 0.016*"people" + 0.015*"apple" + 0.015*"assistant" + '
#   '0.013*"google" + 0.013*"app" + 0.012*"make" + 0.011*"bot" + 0.011*"amazon" '
#   '+ 0.010*"voice" + 0.010*"conversation" + 0.009*"service" + 0.009*"talk" + '
#   '0.008*"call" + 0.008*"information" + 0.008*"siri" + 0.008*"company" + '
#   '0.007*"smart" + 0.007*"thing" + 0.007*"device" + 0.007*"home" + '
#   '0.007*"product" + 0.007*"email" + 0.007*"personal" + 0.006*"message" + '
#   '0.006*"apps" + 0.006*"search" + 0.006*"response" + 0.006*"time" + '
#   '0.006*"phone" + 0.005*"alexa" + 0.005*"work" + 0.005*"feature" + '
#   '0.005*"experience" + 0.004*"end" + 0.004*"year" + 0.004*"language" + '
#   '0.004*"understand" + 0.004*"add" + 0.004*"give"'),
#  (4,
#   '0.049*"data" + 0.040*"company" + 0.019*"business" + 0.013*"customer" + '
#   '0.012*"startup" + 0.012*"million" + 0.010*"product" + 0.010*"technology" + '
#   '0.008*"platform" + 0.008*"big" + 0.008*"year" + 0.008*"digital" + '
#   '0.007*"make" + 0.007*"base" + 0.007*"billion" + 0.007*"market" + '
#   '0.006*"tech" + 0.006*"insight" + 0.006*"money" + 0.006*"intelligence" + '
#   '0.006*"artificial" + 0.006*"scientist" + 0.006*"solution" + 0.006*"machine" '
#   '+ 0.006*"fund" + 0.005*"industry" + 0.005*"venture" + 0.005*"firm" + '
#   '0.005*"ceo" + 0.005*"large" + 0.004*"partner" + 0.004*"service" + '
#   '0.004*"analytics" + 0.004*"investor" + 0.004*"financial" + 0.004*"include" '
#   '+ 0.004*"investment" + 0.004*"grow" + 0.004*"job" + 0.004*"focus"'),
#  (5,
#   '0.033*"car" + 0.021*"drive" + 0.021*"system" + 0.017*"vehicle" + '
#   '0.016*"autonomous" + 0.010*"make" + 0.010*"time" + 0.010*"road" + '
#   '0.010*"human" + 0.008*"driver" + 0.008*"control" + 0.008*"drone" + '
#   '0.007*"map" + 0.007*"area" + 0.006*"fly" + 0.006*"traffic" + 0.005*"sensor" '
#   '+ 0.005*"test" + 0.005*"city" + 0.005*"high" + 0.005*"engineer" + '
#   '0.004*"street" + 0.004*"safety" + 0.004*"information" + 0.004*"mission" + '
#   '0.004*"speed" + 0.004*"environment" + 0.004*"land" + 0.004*"challenge" + '
#   '0.004*"people" + 0.004*"turn" + 0.004*"navigate" + 0.004*"safe" + '
#   '0.004*"change" + 0.004*"space" + 0.004*"uber" + 0.003*"situation" + '
#   '0.003*"automate" + 0.003*"air" + 0.003*"fast"'),
#  (6,
#   '0.026*"chip" + 0.020*"power" + 0.016*"software" + 0.014*"run" + '
#   '0.014*"device" + 0.013*"cloud" + 0.012*"design" + 0.012*"hardware" + '
#   '0.011*"ai" + 0.011*"year" + 0.010*"compute" + 0.010*"time" + '
#   '0.009*"processing" + 0.008*"performance" + 0.008*"intel" + 0.008*"make" + '
#   '0.008*"data" + 0.008*"computer" + 0.007*"nvidia" + 0.007*"camera" + '
#   '0.007*"learn" + 0.007*"network" + 0.007*"developer" + 0.006*"huawei" + '
#   '0.006*"technology" + 0.006*"intelligence" + 0.006*"processor" + '
#   '0.006*"mobile" + 0.006*"application" + 0.006*"built" + 0.006*"server" + '
#   '0.006*"neural" + 0.005*"artificial" + 0.005*"include" + 0.005*"company" + '
#   '0.005*"center" + 0.005*"gpus" + 0.005*"platform" + 0.005*"recognition" + '
#   '0.005*"feature"'),
#  (7,
#   '0.044*"human" + 0.037*"intelligence" + 0.030*"computer" + '
#   '0.022*"artificial" + 0.019*"machine" + 0.016*"program" + 0.016*"ai" + '
#   '0.013*"brain" + 0.012*"make" + 0.012*"world" + 0.010*"question" + '
#   '0.009*"thing" + 0.009*"time" + 0.009*"scientist" + 0.009*"understand" + '
#   '0.008*"problem" + 0.008*"ibm" + 0.008*"work" + 0.008*"mind" + '
#   '0.007*"science" + 0.006*"watson" + 0.006*"write" + 0.006*"knowledge" + '
#   '0.006*"answer" + 0.006*"today" + 0.006*"solve" + 0.005*"sense" + '
#   '0.005*"reason" + 0.005*"figure" + 0.005*"idea" + 0.005*"year" + '
#   '0.004*"common" + 0.004*"field" + 0.004*"thought" + 0.004*"intelligent" + '
#   '0.004*"decade" + 0.004*"turing" + 0.004*"good" + 0.004*"book" + '
#   '0.004*"general"'),
#  (8,
#   '0.033*"model" + 0.029*"image" + 0.028*"learn" + 0.024*"system" + '
#   '0.024*"network" + 0.024*"researcher" + 0.016*"data" + 0.014*"neural" + '
#   '0.013*"train" + 0.011*"paper" + 0.011*"computer" + 0.011*"object" + '
#   '0.010*"training" + 0.008*"machine" + 0.008*"word" + 0.007*"language" + '
#   '0.007*"information" + 0.007*"algorithm" + 0.007*"pattern" + 0.007*"author" '
#   '+ 0.006*"visual" + 0.006*"brain" + 0.006*"task" + 0.006*"process" + '
#   '0.006*"recognition" + 0.006*"layer" + 0.006*"science" + 0.006*"label" + '
#   '0.005*"technique" + 0.005*"neuron" + 0.005*"feature" + 0.005*"processing" + '
#   '0.005*"set" + 0.005*"instance" + 0.005*"node" + 0.005*"method" + '
#   '0.005*"mit" + 0.005*"deep" + 0.004*"research" + 0.004*"work"'),
#  (9,
#   '0.013*"people" + 0.012*"create" + 0.010*"year" + 0.009*"work" + '
#   '0.007*"kind" + 0.007*"thing" + 0.007*"story" + 0.007*"sound" + 0.006*"art" '
#   '+ 0.006*"time" + 0.006*"make" + 0.006*"turn" + 0.006*"start" + 0.005*"back" '
#   '+ 0.005*"technology" + 0.005*"music" + 0.005*"wire" + 0.005*"idea" + '
#   '0.005*"write" + 0.005*"show" + 0.005*"feel" + 0.004*"day" + '
#   '0.004*"experience" + 0.004*"find" + 0.004*"li" + 0.004*"part" + '
#   '0.004*"love" + 0.004*"bit" + 0.003*"live" + 0.003*"life" + 0.003*"future" + '
#   '0.003*"artist" + 0.003*"side" + 0.003*"creative" + 0.003*"style" + '
#   '0.003*"room" + 0.003*"begin" + 0.003*"piece" + 0.003*"talk" + '
#   '0.003*"project"'),
#  (10,
#   '0.054*"robot" + 0.016*"team" + 0.014*"system" + 0.012*"task" + '
#   '0.012*"human" + 0.012*"work" + 0.011*"make" + 0.011*"action" + 0.010*"move" '
#   '+ 0.009*"object" + 0.009*"robotics" + 0.008*"algorithm" + '
#   '0.008*"environment" + 0.008*"researcher" + 0.007*"agent" + 0.007*"robotic" '
#   '+ 0.007*"real" + 0.006*"arm" + 0.006*"part" + 0.006*"pick" + 0.006*"hand" + '
#   '0.005*"paper" + 0.005*"time" + 0.005*"movement" + 0.005*"step" + '
#   '0.005*"control" + 0.005*"give" + 0.005*"shape" + 0.004*"csail" + '
#   '0.004*"project" + 0.004*"world" + 0.004*"motion" + 0.004*"block" + '
#   '0.004*"body" + 0.004*"person" + 0.004*"physical" + 0.004*"reward" + '
#   '0.004*"design" + 0.004*"approach" + 0.004*"university"'),
#  (11,
#   '0.029*"patient" + 0.019*"health" + 0.018*"data" + 0.016*"medical" + '
#   '0.014*"care" + 0.013*"doctor" + 0.011*"model" + 0.009*"disease" + '
#   '0.009*"year" + 0.009*"cancer" + 0.008*"learn" + 0.008*"time" + '
#   '0.008*"hospital" + 0.008*"test" + 0.007*"risk" + 0.007*"predict" + '
#   '0.007*"healthcare" + 0.007*"study" + 0.006*"information" + '
#   '0.006*"treatment" + 0.006*"make" + 0.006*"clinical" + 0.005*"researcher" + '
#   '0.005*"scan" + 0.005*"medicine" + 0.005*"life" + 0.005*"child" + '
#   '0.005*"detect" + 0.005*"condition" + 0.005*"research" + 0.005*"develop" + '
#   '0.005*"improve" + 0.004*"drug" + 0.004*"base" + 0.004*"high" + '
#   '0.004*"heart" + 0.004*"trial" + 0.004*"record" + 0.004*"early" + '
#   '0.004*"diagnosis"'),
#  (12,
#   '0.042*"ai" + 0.033*"technology" + 0.013*"government" + 0.013*"china" + '
#   '0.011*"recognition" + 0.011*"job" + 0.010*"facial" + 0.009*"artificial" + '
#   '0.009*"country" + 0.008*"public" + 0.008*"company" + 0.008*"intelligence" + '
#   '0.007*"year" + 0.007*"policy" + 0.006*"report" + 0.006*"tech" + '
#   '0.006*"american" + 0.006*"include" + 0.006*"worker" + 0.006*"military" + '
#   '0.006*"people" + 0.005*"future" + 0.005*"chinese" + 0.005*"president" + '
#   '0.005*"society" + 0.005*"state" + 0.005*"law" + 0.005*"world" + '
#   '0.005*"national" + 0.005*"police" + 0.004*"concern" + 0.004*"ethical" + '
#   '0.004*"microsoft" + 0.004*"project" + 0.004*"privacy" + 0.004*"development" '
#   '+ 0.004*"agency" + 0.004*"regulation" + 0.004*"lead" + 0.004*"amazon"'),
#  (13,
#   '0.104*"ai" + 0.052*"expo" + 0.024*"locate" + 0.020*"industry" + '
#   '0.018*"leader" + 0.018*"event" + 0.017*"security" + 0.016*"big" + '
#   '0.015*"cloud" + 0.015*"silicon" + 0.015*"london" + 0.015*"valley" + '
#   '0.015*"subject" + 0.015*"interested" + 0.014*"discus" + 0.014*"iot" + '
#   '0.014*"hearing" + 0.014*"case" + 0.014*"attend" + 0.013*"blockchain" + '
#   '0.013*"tech" + 0.013*"cyber" + 0.013*"upcoming" + 0.013*"show" + '
#   '0.013*"amsterdam" + 0.013*"technology" + 0.011*"data" + 0.011*"learn" + '
#   '0.010*"future" + 0.010*"enterprise" + 0.009*"share" + 0.009*"place" + '
#   '0.008*"explore" + 0.008*"uk" + 0.007*"comment" + 0.006*"development" + '
#   '0.005*"provide" + 0.005*"lead" + 0.005*"report" + 0.004*"create"'),
#  (14,
#   '0.047*"mit" + 0.024*"science" + 0.022*"research" + 0.020*"student" + '
#   '0.016*"professor" + 0.014*"work" + 0.012*"computer" + 0.011*"engineering" + '
#   '0.011*"compute" + 0.011*"intelligence" + 0.010*"lab" + 0.009*"group" + '
#   '0.009*"college" + 0.008*"program" + 0.007*"institute" + 0.007*"school" + '
#   '0.007*"department" + 0.007*"include" + 0.007*"technology" + 0.007*"project" '
#   '+ 0.006*"learn" + 0.006*"challenge" + 0.006*"member" + 0.006*"focus" + '
#   '0.006*"university" + 0.005*"community" + 0.005*"field" + 0.005*"faculty" + '
#   '0.005*"graduate" + 0.005*"director" + 0.005*"support" + 0.005*"develop" + '
#   '0.005*"class" + 0.005*"artificial" + 0.004*"collaboration" + '
#   '0.004*"building" + 0.004*"center" + 0.004*"quest" + 0.004*"education" + '
#   '0.004*"opportunity"'),
#  (15,
#   '0.045*"game" + 0.026*"play" + 0.022*"human" + 0.020*"move" + '
#   '0.018*"machine" + 0.015*"player" + 0.015*"learn" + 0.014*"match" + '
#   '0.013*"alphago" + 0.012*"make" + 0.011*"deepmind" + 0.009*"win" + '
#   '0.009*"world" + 0.008*"lee" + 0.007*"ai" + 0.007*"time" + 0.006*"bot" + '
#   '0.006*"top" + 0.006*"system" + 0.006*"team" + 0.006*"beat" + 0.006*"board" '
#   '+ 0.005*"strategy" + 0.005*"sedol" + 0.005*"call" + 0.005*"point" + '
#   '0.005*"year" + 0.005*"researcher" + 0.005*"played" + 0.005*"chess" + '
#   '0.005*"million" + 0.004*"reinforcement" + 0.004*"hour" + 0.004*"hand" + '
#   '0.004*"contest" + 0.004*"lose" + 0.004*"month" + 0.004*"end" + '
#   '0.004*"chance" + 0.003*"press"'),
#  (16,
#   '0.049*"google" + 0.047*"learn" + 0.032*"ai" + 0.025*"company" + '
#   '0.024*"machine" + 0.021*"deep" + 0.017*"facebook" + 0.016*"work" + '
#   '0.015*"neural" + 0.012*"microsoft" + 0.011*"network" + 0.011*"system" + '
#   '0.009*"world" + 0.009*"research" + 0.009*"build" + 0.008*"data" + '
#   '0.008*"photo" + 0.007*"open" + 0.007*"call" + 0.007*"software" + '
#   '0.007*"researcher" + 0.007*"project" + 0.007*"service" + 0.007*"net" + '
#   '0.006*"search" + 0.006*"technology" + 0.006*"lab" + 0.006*"recognize" + '
#   '0.006*"big" + 0.006*"language" + 0.005*"artificial" + 0.005*"building" + '
#   '0.005*"year" + 0.005*"openai" + 0.005*"engine" + 0.005*"team" + '
#   '0.005*"source" + 0.005*"analyze" + 0.005*"task" + 0.005*"internet"')]


####################################################
# ####################################################
# #####                   15 topics
# ####################################################
# ####################################################
#
# Coherence Score:  0.49282630003274963
# ----------------Topics for LDA-Mallet ----------------------------
# [(0, Research
#   '0.047*"mit" + 0.024*"science" + 0.022*"research" + 0.019*"student" + '
#   '0.017*"professor" + 0.015*"computer" + 0.014*"work" + 0.014*"intelligence" '
#   '+ 0.013*"engineering" + 0.011*"compute" + 0.010*"group" + 0.010*"lab" + '
#   '0.008*"college" + 0.008*"institute" + 0.008*"program" + 0.008*"school" + '
#   '0.008*"department" + 0.007*"project" + 0.007*"learn" + 0.007*"university" + '
#   '0.006*"challenge" + 0.006*"field" + 0.006*"include" + 0.005*"graduate" + '
#   '0.005*"center" + 0.005*"member" + 0.005*"focus" + 0.005*"community" + '
#   '0.005*"faculty" + 0.005*"machine" + 0.005*"director" + 0.005*"class" + '
#   '0.005*"artificial" + 0.005*"develop" + 0.005*"support" + '
#   '0.005*"collaboration" + 0.004*"electrical" + 0.004*"medium" + 0.004*"quest" '
#   '+ 0.004*"advance"'),
#  (1, ai in big company (silicon valley)
#   '0.097*"ai" + 0.051*"expo" + 0.023*"locate" + 0.020*"industry" + '
#   '0.017*"event" + 0.017*"leader" + 0.016*"security" + 0.016*"data" + '
#   '0.016*"cloud" + 0.015*"big" + 0.015*"subject" + 0.015*"valley" + '
#   '0.015*"silicon" + 0.014*"london" + 0.014*"interested" + 0.014*"technology" '
#   '+ 0.014*"discus" + 0.014*"iot" + 0.014*"hearing" + 0.013*"attend" + '
#   '0.013*"blockchain" + 0.013*"case" + 0.013*"show" + 0.013*"cyber" + '
#   '0.013*"upcoming" + 0.012*"tech" + 0.012*"amsterdam" + 0.010*"future" + '
#   '0.010*"learn" + 0.010*"place" + 0.009*"enterprise" + 0.009*"share" + '
#   '0.008*"explore" + 0.008*"uk" + 0.006*"comment" + 0.005*"business" + '
#   '0.005*"percent" + 0.005*"development" + 0.004*"report" + 0.004*"provide"'),
#  (2, ai in voice answer
#   '0.018*"people" + 0.009*"bot" + 0.009*"assistant" + 0.009*"work" + '
#   '0.009*"conversation" + 0.008*"voice" + 0.008*"talk" + 0.007*"sound" + '
#   '0.007*"create" + 0.007*"siri" + 0.006*"call" + 0.006*"make" + 0.006*"thing" '
#   '+ 0.006*"amazon" + 0.005*"back" + 0.005*"user" + 0.005*"email" + '
#   '0.005*"message" + 0.005*"time" + 0.005*"language" + 0.004*"friend" + '
#   '0.004*"year" + 0.004*"alexa" + 0.004*"give" + 0.004*"place" + '
#   '0.004*"response" + 0.004*"feel" + 0.004*"show" + 0.004*"record" + '
#   '0.004*"music" + 0.004*"person" + 0.004*"start" + 0.004*"turn" + 0.004*"art" '
#   '+ 0.004*"home" + 0.004*"word" + 0.004*"virtual" + 0.004*"good" + '
#   '0.004*"room" + 0.004*"story"'),
#  (3,  ai in automeded cars
#   '0.029*"car" + 0.022*"system" + 0.020*"drive" + 0.015*"vehicle" + '
#   '0.014*"autonomous" + 0.012*"human" + 0.011*"make" + 0.011*"time" + '
#   '0.009*"road" + 0.008*"algorithm" + 0.008*"control" + 0.008*"driver" + '
#   '0.007*"drone" + 0.007*"map" + 0.007*"test" + 0.007*"environment" + '
#   '0.005*"fly" + 0.005*"decision" + 0.005*"sensor" + 0.005*"traffic" + '
#   '0.005*"challenge" + 0.005*"people" + 0.004*"engineer" + 0.004*"path" + '
#   '0.004*"area" + 0.004*"information" + 0.004*"agent" + 0.004*"safety" + '
#   '0.004*"situation" + 0.004*"team" + 0.004*"street" + 0.004*"planning" + '
#   '0.004*"travel" + 0.004*"city" + 0.004*"speed" + 0.004*"navigate" + '
#   '0.003*"high" + 0.003*"land" + 0.003*"safe" + 0.003*"design"'),
#  (4, ai job
#   '0.048*"ai" + 0.032*"technology" + 0.014*"job" + 0.013*"government" + '
#   '0.012*"china" + 0.010*"intelligence" + 0.010*"artificial" + 0.009*"country" '
#   '+ 0.009*"future" + 0.008*"people" + 0.007*"report" + 0.007*"society" + '
#   '0.006*"policy" + 0.006*"work" + 0.006*"public" + 0.006*"president" + '
#   '0.006*"development" + 0.006*"world" + 0.006*"worker" + 0.005*"military" + '
#   '0.005*"american" + 0.005*"make" + 0.005*"global" + 0.005*"ethical" + '
#   '0.005*"project" + 0.005*"force" + 0.005*"year" + 0.005*"national" + '
#   '0.005*"issue" + 0.005*"microsoft" + 0.005*"potential" + 0.004*"lead" + '
#   '0.004*"state" + 0.004*"task" + 0.004*"concern" + 0.004*"ethic" + '
#   '0.004*"board" + 0.004*"regulation" + 0.004*"defense" + 0.004*"economic"'),
#  (5,  ai in business,
#   '0.053*"company" + 0.018*"technology" + 0.017*"year" + 0.017*"ai" + '
#   '0.015*"million" + 0.014*"startup" + 0.013*"tech" + 0.010*"ceo" + '
#   '0.009*"work" + 0.009*"billion" + 0.008*"build" + 0.008*"software" + '
#   '0.008*"openai" + 0.008*"artificial" + 0.008*"make" + 0.008*"world" + '
#   '0.007*"business" + 0.007*"intelligence" + 0.006*"big" + 0.006*"founder" + '
#   '0.006*"top" + 0.006*"money" + 0.006*"include" + 0.006*"fund" + '
#   '0.005*"market" + 0.005*"open" + 0.005*"musk" + 0.005*"venture" + '
#   '0.005*"project" + 0.005*"share" + 0.005*"base" + 0.005*"firm" + '
#   '0.004*"grow" + 0.004*"talent" + 0.004*"hire" + 0.004*"start" + 0.004*"li" + '
#   '0.004*"investor" + 0.004*"industry" + 0.004*"investment"'),
#  (6, a in health
#   '0.025*"patient" + 0.020*"data" + 0.016*"health" + 0.014*"medical" + '
#   '0.012*"care" + 0.011*"doctor" + 0.010*"model" + 0.009*"risk" + '
#   '0.008*"disease" + 0.008*"year" + 0.008*"cancer" + 0.007*"study" + '
#   '0.007*"test" + 0.007*"hospital" + 0.007*"learn" + 0.007*"time" + '
#   '0.006*"percent" + 0.006*"result" + 0.006*"predict" + 0.006*"treatment" + '
#   '0.005*"make" + 0.005*"healthcare" + 0.005*"life" + 0.005*"high" + '
#   '0.005*"information" + 0.005*"score" + 0.005*"child" + 0.005*"clinical" + '
#   '0.005*"medicine" + 0.005*"algorithm" + 0.005*"improve" + 0.005*"scan" + '
#   '0.004*"drug" + 0.004*"researcher" + 0.004*"people" + 0.004*"condition" + '
#   '0.004*"record" + 0.004*"machine" + 0.004*"start" + 0.004*"trial"'),
#  (7, ai in answers
#   '0.046*"human" + 0.024*"intelligence" + 0.021*"computer" + 0.017*"machine" + '
#   '0.015*"make" + 0.014*"artificial" + 0.013*"program" + 0.013*"thing" + '
#   '0.012*"ai" + 0.011*"question" + 0.011*"problem" + 0.010*"people" + '
#   '0.009*"write" + 0.008*"understand" + 0.008*"work" + 0.007*"world" + '
#   '0.006*"brain" + 0.006*"mind" + 0.006*"reason" + 0.006*"answer" + '
#   '0.006*"time" + 0.006*"year" + 0.006*"solve" + 0.005*"ibm" + 0.005*"good" + '
#   '0.005*"scientist" + 0.005*"test" + 0.005*"watson" + 0.005*"life" + '
#   '0.004*"fact" + 0.004*"kind" + 0.004*"knowledge" + 0.004*"idea" + '
#   '0.004*"decision" + 0.004*"science" + 0.004*"book" + 0.004*"give" + '
#   '0.004*"wire" + 0.004*"thought" + 0.004*"today"'),
#  (8, ai in face derection
#   '0.017*"algorithm" + 0.016*"video" + 0.015*"recognition" + 0.013*"facial" + '
#   '0.012*"face" + 0.011*"people" + 0.011*"news" + 0.011*"content" + '
#   '0.009*"facebook" + 0.008*"percent" + 0.008*"fake" + 0.008*"bias" + '
#   '0.007*"system" + 0.006*"photo" + 0.006*"tool" + 0.006*"post" + '
#   '0.006*"medium" + 0.006*"identify" + 0.006*"report" + 0.006*"law" + '
#   '0.005*"social" + 0.005*"online" + 0.005*"problem" + 0.005*"police" + '
#   '0.005*"show" + 0.005*"woman" + 0.005*"public" + 0.005*"call" + '
#   '0.005*"group" + 0.005*"software" + 0.004*"test" + 0.004*"detect" + '
#   '0.004*"amazon" + 0.004*"company" + 0.004*"black" + 0.004*"found" + '
#   '0.004*"person" + 0.004*"review" + 0.004*"issue" + 0.004*"ad"'),
#  (9, dnn in face recognition
#   '0.030*"model" + 0.029*"image" + 0.026*"system" + 0.025*"researcher" + '
#   '0.024*"learn" + 0.021*"network" + 0.013*"train" + 0.013*"neural" + '
#   '0.011*"computer" + 0.011*"paper" + 0.011*"data" + 0.010*"training" + '
#   '0.010*"object" + 0.010*"machine" + 0.009*"language" + 0.008*"word" + '
#   '0.007*"algorithm" + 0.007*"pattern" + 0.007*"set" + 0.007*"brain" + '
#   '0.006*"author" + 0.006*"video" + 0.006*"information" + 0.006*"visual" + '
#   '0.005*"label" + 0.005*"work" + 0.005*"task" + 0.005*"layer" + '
#   '0.005*"feature" + 0.005*"method" + 0.005*"science" + 0.005*"instance" + '
#   '0.005*"processing" + 0.005*"technique" + 0.004*"neuron" + 0.004*"human" + '
#   '0.004*"recognition" + 0.004*"identify" + 0.004*"vision" + '
#   '0.004*"recognize"'),
#  (10,ai in smart phone
#   '0.039*"data" + 0.016*"product" + 0.015*"apple" + 0.015*"device" + '
#   '0.015*"user" + 0.014*"customer" + 0.013*"platform" + 0.012*"service" + '
#   '0.012*"company" + 0.011*"make" + 0.009*"feature" + 0.008*"business" + '
#   '0.008*"time" + 0.008*"base" + 0.007*"power" + 0.007*"year" + '
#   '0.007*"machine" + 0.006*"app" + 0.006*"mobile" + 0.006*"information" + '
#   '0.006*"offer" + 0.006*"smart" + 0.006*"developer" + 0.006*"consumer" + '
#   '0.006*"experience" + 0.006*"launch" + 0.006*"software" + '
#   '0.006*"performance" + 0.005*"deliver" + 0.005*"digital" + 0.005*"store" + '
#   '0.005*"provide" + 0.005*"huawei" + 0.005*"big" + 0.005*"insight" + '
#   '0.005*"improve" + 0.005*"camera" + 0.004*"thing" + 0.004*"easy" + '
#   '0.004*"access"'),
#  (11, ai in robot
#   '0.049*"robot" + 0.018*"team" + 0.013*"system" + 0.012*"work" + 0.011*"task" '
#   '+ 0.009*"make" + 0.009*"object" + 0.008*"robotics" + 0.008*"move" + '
#   '0.007*"researcher" + 0.007*"robotic" + 0.007*"real" + 0.007*"computer" + '
#   '0.006*"design" + 0.006*"human" + 0.006*"hand" + 0.006*"control" + '
#   '0.006*"arm" + 0.006*"part" + 0.005*"action" + 0.005*"shape" + '
#   '0.005*"project" + 0.005*"algorithm" + 0.005*"time" + 0.005*"movement" + '
#   '0.005*"create" + 0.005*"pick" + 0.005*"step" + 0.005*"paper" + '
#   '0.005*"environment" + 0.004*"body" + 0.004*"university" + 0.004*"motion" + '
#   '0.004*"csail" + 0.004*"physical" + 0.004*"developed" + 0.004*"approach" + '
#   '0.004*"camera" + 0.004*"block" + 0.003*"person"'),
#  (12, ai research
#   '0.020*"data" + 0.011*"algorithm" + 0.011*"researcher" + 0.010*"material" + '
#   '0.010*"process" + 0.010*"design" + 0.009*"problem" + 0.007*"time" + '
#   '0.007*"system" + 0.007*"science" + 0.007*"complex" + 0.007*"energy" + '
#   '0.006*"research" + 0.006*"change" + 0.006*"make" + 0.006*"scientist" + '
#   '0.005*"set" + 0.005*"find" + 0.005*"work" + 0.005*"high" + 0.005*"model" + '
#   '0.005*"structure" + 0.005*"number" + 0.005*"machine" + 0.005*"large" + '
#   '0.004*"learn" + 0.004*"cell" + 0.004*"small" + 0.004*"produce" + '
#   '0.004*"technique" + 0.004*"approach" + 0.004*"space" + 0.004*"mit" + '
#   '0.004*"engineering" + 0.004*"paper" + 0.004*"property" + 0.004*"graph" + '
#   '0.004*"generate" + 0.004*"give" + 0.004*"call"'),
#  (13, ai in game
#   '0.040*"game" + 0.024*"play" + 0.022*"human" + 0.020*"machine" + '
#   '0.020*"learn" + 0.019*"move" + 0.014*"player" + 0.013*"match" + '
#   '0.012*"alphago" + 0.011*"world" + 0.010*"deepmind" + 0.009*"make" + '
#   '0.009*"ai" + 0.008*"win" + 0.008*"time" + 0.007*"team" + 0.007*"system" + '
#   '0.007*"lee" + 0.006*"researcher" + 0.006*"top" + 0.006*"reinforcement" + '
#   '0.005*"call" + 0.005*"beat" + 0.005*"point" + 0.005*"sedol" + '
#   '0.005*"played" + 0.005*"complex" + 0.005*"action" + 0.004*"bot" + '
#   '0.004*"million" + 0.004*"hand" + 0.004*"chess" + 0.004*"strategy" + '
#   '0.004*"show" + 0.004*"board" + 0.004*"year" + 0.004*"google" + 0.004*"hour" '
#   '+ 0.004*"contest" + 0.003*"built"'),
#  (14, ai in chip
#   '0.053*"google" + 0.045*"learn" + 0.022*"ai" + 0.022*"deep" + '
#   '0.019*"machine" + 0.018*"neural" + 0.017*"facebook" + 0.016*"company" + '
#   '0.015*"network" + 0.015*"chip" + 0.013*"work" + 0.010*"microsoft" + '
#   '0.010*"software" + 0.010*"data" + 0.009*"photo" + 0.008*"search" + '
#   '0.008*"call" + 0.008*"build" + 0.008*"system" + 0.008*"research" + '
#   '0.007*"artificial" + 0.007*"service" + 0.007*"world" + 0.007*"run" + '
#   '0.007*"net" + 0.006*"technology" + 0.006*"power" + 0.006*"researcher" + '
#   '0.006*"source" + 0.006*"compute" + 0.006*"brain" + 0.006*"project" + '
#   '0.006*"internet" + 0.005*"recognize" + 0.005*"hardware" + 0.005*"open" + '
#   '0.005*"understand" + 0.005*"building" + 0.005*"include" + 0.005*"year"')]

####################################################
####################################################
#####                   19 topics
####################################################
####################################################
#----------------Perplexity and Coherence Score for LDA-Mallet----------------------------
#
# Coherence Score:  0.49263003593256754
# ----------------Topics for LDA-Mallet ----------------------------
# [(0,
#   '0.033*"image" + 0.031*"model" + 0.030*"learn" + 0.026*"network" + '
#   '0.025*"system" + 0.024*"researcher" + 0.016*"neural" + 0.015*"data" + '
#   '0.013*"train" + 0.012*"object" + 0.012*"computer" + 0.012*"training" + '
#   '0.011*"machine" + 0.010*"paper" + 0.009*"brain" + 0.008*"language" + '
#   '0.007*"word" + 0.007*"pattern" + 0.007*"feature" + 0.007*"task" + '
#   '0.007*"visual" + 0.006*"information" + 0.006*"recognition" + '
#   '0.006*"identify" + 0.006*"label" + 0.006*"layer" + 0.006*"neuron" + '
#   '0.005*"process" + 0.005*"processing" + 0.005*"recognize" + 0.005*"deep" + '
#   '0.005*"science" + 0.005*"algorithm" + 0.005*"video" + 0.005*"author" + '
#   '0.005*"result" + 0.005*"percent" + 0.005*"work" + 0.004*"method" + '
#   '0.004*"set"'),
#  (1,
#   '0.037*"car" + 0.024*"drive" + 0.023*"system" + 0.019*"vehicle" + '
#   '0.017*"autonomous" + 0.011*"road" + 0.009*"driver" + 0.009*"control" + '
#   '0.009*"drone" + 0.008*"map" + 0.008*"time" + 0.008*"make" + 0.008*"test" + '
#   '0.007*"environment" + 0.007*"fly" + 0.006*"human" + 0.006*"traffic" + '
#   '0.006*"sensor" + 0.006*"team" + 0.005*"area" + 0.005*"change" + '
#   '0.005*"city" + 0.005*"mission" + 0.005*"safety" + 0.004*"algorithm" + '
#   '0.004*"engineer" + 0.004*"air" + 0.004*"safe" + 0.004*"land" + 0.004*"high" '
#   '+ 0.004*"street" + 0.004*"plan" + 0.004*"challenge" + 0.004*"develop" + '
#   '0.004*"track" + 0.004*"route" + 0.004*"speed" + 0.004*"automate" + '
#   '0.004*"space" + 0.004*"navigate"'),
#  (2,
#   '0.020*"technology" + 0.019*"recognition" + 0.017*"facial" + '
#   '0.017*"algorithm" + 0.012*"company" + 0.012*"people" + 0.012*"percent" + '
#   '0.011*"bias" + 0.011*"face" + 0.010*"law" + 0.010*"system" + 0.009*"public" '
#   '+ 0.008*"include" + 0.007*"group" + 0.007*"woman" + 0.007*"microsoft" + '
#   '0.007*"call" + 0.007*"privacy" + 0.007*"tech" + 0.007*"police" + '
#   '0.006*"decision" + 0.006*"concern" + 0.006*"test" + 0.006*"amazon" + '
#   '0.006*"government" + 0.006*"individual" + 0.005*"city" + '
#   '0.005*"surveillance" + 0.005*"found" + 0.005*"report" + 0.005*"study" + '
#   '0.005*"tool" + 0.005*"black" + 0.004*"gender" + 0.004*"risk" + '
#   '0.004*"state" + 0.004*"society" + 0.004*"bill" + 0.004*"issue" + '
#   '0.004*"rule"'),
#  (3,
#   '0.030*"patient" + 0.019*"health" + 0.019*"data" + 0.017*"medical" + '
#   '0.014*"care" + 0.013*"doctor" + 0.011*"model" + 0.010*"disease" + '
#   '0.009*"cancer" + 0.009*"risk" + 0.008*"hospital" + 0.008*"study" + '
#   '0.008*"test" + 0.008*"healthcare" + 0.007*"year" + 0.007*"information" + '
#   '0.007*"predict" + 0.007*"treatment" + 0.006*"time" + 0.006*"life" + '
#   '0.006*"clinical" + 0.006*"percent" + 0.006*"scan" + 0.006*"medicine" + '
#   '0.005*"condition" + 0.005*"drug" + 0.005*"detect" + 0.005*"child" + '
#   '0.005*"develop" + 0.005*"high" + 0.005*"make" + 0.005*"researcher" + '
#   '0.005*"learn" + 0.005*"trial" + 0.004*"result" + 0.004*"record" + '
#   '0.004*"reduce" + 0.004*"research" + 0.004*"base" + 0.004*"improve"'),
#  (4,
#   '0.024*"video" + 0.016*"news" + 0.015*"content" + 0.014*"facebook" + '
#   '0.012*"fake" + 0.011*"social" + 0.011*"online" + 0.011*"algorithm" + '
#   '0.010*"people" + 0.009*"post" + 0.009*"user" + 0.008*"create" + '
#   '0.008*"write" + 0.008*"text" + 0.008*"photo" + 0.008*"medium" + '
#   '0.007*"story" + 0.007*"tool" + 0.007*"twitter" + 0.007*"identify" + '
#   '0.007*"problem" + 0.006*"article" + 0.006*"make" + 0.006*"platform" + '
#   '0.006*"report" + 0.006*"work" + 0.006*"youtube" + 0.006*"speech" + '
#   '0.005*"ad" + 0.005*"generate" + 0.005*"deepfakes" + 0.005*"internet" + '
#   '0.005*"attack" + 0.004*"wire" + 0.004*"person" + 0.004*"account" + '
#   '0.004*"software" + 0.004*"automate" + 0.004*"flag" + 0.004*"remove"'),
#  (5,
#   '0.074*"human" + 0.033*"intelligence" + 0.030*"ai" + 0.029*"machine" + '
#   '0.020*"make" + 0.017*"artificial" + 0.016*"question" + 0.012*"problem" + '
#   '0.010*"answer" + 0.009*"decision" + 0.009*"computer" + 0.009*"world" + '
#   '0.009*"understand" + 0.008*"reason" + 0.008*"test" + 0.008*"ibm" + '
#   '0.007*"watson" + 0.007*"thing" + 0.007*"mind" + 0.007*"solve" + '
#   '0.006*"time" + 0.006*"good" + 0.006*"form" + 0.005*"intelligent" + '
#   '0.005*"knowledge" + 0.005*"ability" + 0.005*"turing" + 0.004*"thought" + '
#   '0.004*"common" + 0.004*"today" + 0.004*"real" + 0.004*"brain" + '
#   '0.004*"level" + 0.004*"give" + 0.004*"complex" + 0.004*"general" + '
#   '0.004*"set" + 0.004*"important" + 0.003*"fact" + 0.003*"read"'),
#  (6,
#   '0.050*"game" + 0.028*"play" + 0.025*"move" + 0.023*"human" + 0.017*"player" '
#   '+ 0.017*"learn" + 0.016*"machine" + 0.015*"match" + 0.015*"alphago" + '
#   '0.013*"deepmind" + 0.010*"win" + 0.009*"world" + 0.009*"lee" + 0.009*"make" '
#   '+ 0.008*"top" + 0.007*"time" + 0.007*"system" + 0.007*"beat" + 0.006*"team" '
#   '+ 0.006*"million" + 0.006*"sedol" + 0.006*"ai" + 0.006*"strategy" + '
#   '0.005*"chess" + 0.005*"bot" + 0.005*"played" + 0.005*"point" + '
#   '0.005*"reinforcement" + 0.005*"board" + 0.005*"year" + 0.005*"call" + '
#   '0.004*"contest" + 0.004*"hand" + 0.004*"action" + 0.004*"complex" + '
#   '0.004*"hour" + 0.004*"minute" + 0.004*"professional" + 0.004*"champion" + '
#   '0.004*"month"'),
#  (7,
#   '0.032*"people" + 0.021*"make" + 0.020*"thing" + 0.019*"work" + 0.017*"time" '
#   '+ 0.015*"start" + 0.014*"year" + 0.012*"day" + 0.012*"back" + 0.011*"talk" '
#   '+ 0.010*"good" + 0.009*"give" + 0.009*"big" + 0.008*"turn" + 0.008*"put" + '
#   '0.008*"end" + 0.008*"idea" + 0.007*"kind" + 0.007*"point" + 0.007*"happen" '
#   '+ 0.007*"interest" + 0.006*"find" + 0.006*"begin" + 0.006*"bit" + '
#   '0.006*"figure" + 0.005*"call" + 0.005*"hard" + 0.005*"ago" + 0.005*"place" '
#   '+ 0.005*"run" + 0.005*"realize" + 0.005*"wire" + 0.005*"build" + '
#   '0.005*"early" + 0.004*"moment" + 0.004*"head" + 0.004*"feel" + 0.004*"hand" '
#   '+ 0.004*"sort" + 0.004*"life"'),
#  (8,
#   '0.099*"ai" + 0.057*"expo" + 0.026*"locate" + 0.020*"industry" + '
#   '0.020*"event" + 0.019*"leader" + 0.019*"cloud" + 0.018*"security" + '
#   '0.017*"silicon" + 0.017*"valley" + 0.016*"london" + 0.016*"interested" + '
#   '0.016*"subject" + 0.016*"discus" + 0.016*"iot" + 0.016*"big" + '
#   '0.016*"hearing" + 0.015*"show" + 0.015*"attend" + 0.015*"blockchain" + '
#   '0.014*"case" + 0.014*"cyber" + 0.014*"upcoming" + 0.014*"tech" + '
#   '0.014*"amsterdam" + 0.011*"enterprise" + 0.010*"place" + 0.010*"share" + '
#   '0.010*"future" + 0.009*"data" + 0.009*"uk" + 0.008*"explore" + '
#   '0.008*"technology" + 0.008*"learn" + 0.007*"comment" + 0.005*"huawei" + '
#   '0.005*"provide" + 0.005*"development" + 0.005*"report" + 0.004*"percent"'),
#  (9,
#   '0.028*"chip" + 0.020*"power" + 0.017*"software" + 0.015*"run" + '
#   '0.014*"design" + 0.013*"device" + 0.012*"compute" + 0.011*"hardware" + '
#   '0.011*"cloud" + 0.011*"year" + 0.010*"time" + 0.010*"network" + '
#   '0.009*"processing" + 0.009*"intel" + 0.008*"make" + 0.008*"nvidia" + '
#   '0.008*"performance" + 0.007*"learn" + 0.007*"ai" + 0.007*"camera" + '
#   '0.007*"mobile" + 0.007*"processor" + 0.007*"artificial" + 0.006*"data" + '
#   '0.006*"build" + 0.006*"technology" + 0.006*"application" + 0.006*"server" + '
#   '0.006*"built" + 0.006*"center" + 0.006*"computer" + 0.006*"developer" + '
#   '0.006*"include" + 0.005*"energy" + 0.005*"neural" + 0.005*"gpus" + '
#   '0.005*"deep" + 0.005*"faster" + 0.005*"company" + 0.005*"platform"'),
#  (10,
#   '0.066*"data" + 0.038*"company" + 0.021*"business" + 0.015*"customer" + '
#   '0.012*"million" + 0.011*"product" + 0.011*"startup" + 0.010*"technology" + '
#   '0.010*"base" + 0.009*"machine" + 0.009*"learn" + 0.009*"platform" + '
#   '0.007*"market" + 0.007*"industry" + 0.007*"insight" + 0.007*"intelligence" '
#   '+ 0.007*"firm" + 0.007*"solution" + 0.007*"big" + 0.006*"process" + '
#   '0.006*"billion" + 0.006*"venture" + 0.006*"fund" + 0.006*"service" + '
#   '0.005*"year" + 0.005*"digital" + 0.005*"partner" + 0.005*"money" + '
#   '0.005*"analysis" + 0.005*"automate" + 0.005*"ceo" + 0.005*"scientist" + '
#   '0.005*"application" + 0.005*"focus" + 0.005*"analytics" + 0.005*"team" + '
#   '0.005*"make" + 0.004*"large" + 0.004*"investor" + 0.004*"financial"'),
#  (11,
#   '0.054*"mit" + 0.025*"science" + 0.022*"research" + 0.022*"student" + '
#   '0.017*"professor" + 0.015*"engineering" + 0.013*"work" + '
#   '0.012*"intelligence" + 0.012*"compute" + 0.012*"group" + 0.010*"college" + '
#   '0.010*"computer" + 0.009*"lab" + 0.009*"department" + 0.008*"institute" + '
#   '0.008*"school" + 0.007*"learn" + 0.007*"project" + 0.007*"include" + '
#   '0.007*"challenge" + 0.007*"graduate" + 0.006*"member" + 0.006*"community" + '
#   '0.006*"technology" + 0.006*"faculty" + 0.006*"center" + 0.005*"focus" + '
#   '0.005*"machine" + 0.005*"director" + 0.005*"advance" + 0.005*"quest" + '
#   '0.005*"field" + 0.005*"support" + 0.005*"electrical" + '
#   '0.005*"collaboration" + 0.005*"area" + 0.005*"design" + 0.004*"medium" + '
#   '0.004*"develop" + 0.004*"impact"'),
#  (12,
#   '0.041*"computer" + 0.032*"program" + 0.021*"university" + 0.020*"scientist" '
#   '+ 0.018*"work" + 0.017*"science" + 0.015*"year" + 0.014*"learn" + '
#   '0.011*"code" + 0.010*"field" + 0.009*"artificial" + 0.009*"research" + '
#   '0.008*"write" + 0.008*"lab" + 0.007*"system" + 0.007*"intelligence" + '
#   '0.006*"brain" + 0.006*"time" + 0.006*"teach" + 0.006*"world" + 0.006*"high" '
#   '+ 0.006*"study" + 0.006*"stanford" + 0.005*"change" + 0.005*"found" + '
#   '0.005*"school" + 0.005*"berkeley" + 0.005*"decade" + 0.004*"child" + '
#   '0.004*"grow" + 0.004*"problem" + 0.004*"concept" + 0.004*"team" + '
#   '0.004*"idea" + 0.004*"long" + 0.004*"inspire" + 0.004*"today" + '
#   '0.004*"language" + 0.004*"professor" + 0.004*"john"'),
#  (13,
#   '0.018*"create" + 0.013*"sound" + 0.011*"show" + 0.010*"art" + '
#   '0.009*"technology" + 0.008*"work" + 0.008*"make" + 0.008*"project" + '
#   '0.007*"people" + 0.007*"music" + 0.006*"reality" + 0.006*"image" + '
#   '0.006*"eye" + 0.005*"real" + 0.005*"style" + 0.005*"track" + 0.005*"artist" '
#   '+ 0.005*"record" + 0.005*"emotion" + 0.005*"creative" + 0.004*"vision" + '
#   '0.004*"virtual" + 0.004*"kind" + 0.004*"thing" + 0.004*"software" + '
#   '0.004*"generate" + 0.004*"world" + 0.004*"feel" + 0.004*"expression" + '
#   '0.004*"film" + 0.004*"star" + 0.004*"movie" + 0.004*"face" + 0.004*"wire" + '
#   '0.004*"video" + 0.004*"design" + 0.004*"picture" + 0.004*"call" + '
#   '0.003*"idea" + 0.003*"space"'),
#  (14,
#   '0.017*"algorithm" + 0.016*"researcher" + 0.015*"data" + 0.013*"system" + '
#   '0.012*"material" + 0.008*"paper" + 0.008*"process" + 0.008*"problem" + '
#   '0.008*"set" + 0.007*"design" + 0.007*"time" + 0.007*"science" + 0.006*"mit" '
#   '+ 0.006*"structure" + 0.006*"high" + 0.006*"find" + 0.006*"technique" + '
#   '0.006*"engineering" + 0.006*"complex" + 0.006*"model" + 0.006*"research" + '
#   '0.006*"work" + 0.005*"approach" + 0.005*"point" + 0.005*"number" + '
#   '0.005*"method" + 0.005*"small" + 0.005*"property" + 0.004*"graph" + '
#   '0.004*"step" + 0.004*"produce" + 0.004*"represent" + 0.004*"author" + '
#   '0.004*"professor" + 0.004*"make" + 0.004*"result" + 0.004*"cell" + '
#   '0.004*"molecule" + 0.004*"input" + 0.004*"space"'),
#  (15,
#   '0.023*"user" + 0.020*"apple" + 0.019*"assistant" + 0.017*"google" + '
#   '0.015*"app" + 0.014*"bot" + 0.013*"voice" + 0.012*"amazon" + '
#   '0.011*"conversation" + 0.011*"company" + 0.011*"service" + 0.010*"people" + '
#   '0.010*"device" + 0.009*"siri" + 0.009*"information" + 0.009*"smart" + '
#   '0.008*"make" + 0.008*"product" + 0.008*"search" + 0.008*"email" + '
#   '0.007*"apps" + 0.007*"phone" + 0.007*"home" + 0.006*"message" + '
#   '0.006*"call" + 0.006*"alexa" + 0.006*"response" + 0.006*"year" + '
#   '0.006*"personal" + 0.006*"launch" + 0.005*"speech" + 0.005*"feature" + '
#   '0.005*"language" + 0.005*"talk" + 0.004*"chat" + 0.004*"person" + '
#   '0.004*"experience" + 0.004*"interface" + 0.004*"store" + 0.004*"add"'),
#  (16,
#   '0.060*"robot" + 0.019*"team" + 0.015*"system" + 0.014*"task" + '
#   '0.011*"human" + 0.011*"work" + 0.010*"action" + 0.010*"robotics" + '
#   '0.010*"object" + 0.009*"make" + 0.009*"move" + 0.008*"algorithm" + '
#   '0.008*"robotic" + 0.008*"researcher" + 0.007*"agent" + 0.007*"control" + '
#   '0.007*"real" + 0.007*"arm" + 0.007*"environment" + 0.006*"part" + '
#   '0.006*"shape" + 0.006*"hand" + 0.006*"csail" + 0.006*"movement" + '
#   '0.005*"paper" + 0.005*"pick" + 0.005*"step" + 0.005*"physical" + '
#   '0.005*"motion" + 0.005*"project" + 0.004*"world" + 0.004*"developed" + '
#   '0.004*"professor" + 0.004*"approach" + 0.004*"block" + 0.004*"body" + '
#   '0.004*"university" + 0.004*"include" + 0.004*"mit" + 0.004*"signal"'),
#  (17,
#   '0.052*"google" + 0.045*"learn" + 0.029*"ai" + 0.028*"company" + '
#   '0.025*"machine" + 0.022*"deep" + 0.019*"facebook" + 0.015*"neural" + '
#   '0.014*"work" + 0.011*"microsoft" + 0.011*"system" + 0.011*"network" + '
#   '0.010*"world" + 0.009*"research" + 0.009*"build" + 0.009*"software" + '
#   '0.009*"photo" + 0.008*"data" + 0.008*"researcher" + 0.008*"open" + '
#   '0.008*"call" + 0.007*"net" + 0.007*"artificial" + 0.007*"project" + '
#   '0.007*"technology" + 0.007*"service" + 0.006*"intelligence" + 0.006*"big" + '
#   '0.006*"search" + 0.006*"building" + 0.006*"recognize" + 0.005*"openai" + '
#   '0.005*"understand" + 0.005*"include" + 0.005*"language" + 0.005*"giant" + '
#   '0.005*"source" + 0.005*"internet" + 0.005*"analyze" + 0.005*"lab"'),
#  (18,
#   '0.064*"ai" + 0.028*"technology" + 0.015*"job" + 0.015*"china" + '
#   '0.014*"artificial" + 0.012*"intelligence" + 0.010*"government" + '
#   '0.009*"country" + 0.009*"future" + 0.009*"world" + 0.008*"lead" + '
#   '0.007*"research" + 0.007*"policy" + 0.007*"president" + 0.007*"development" '
#   '+ 0.006*"worker" + 0.006*"year" + 0.006*"military" + 0.006*"report" + '
#   '0.006*"chinese" + 0.006*"global" + 0.006*"national" + 0.005*"american" + '
#   '0.005*"tech" + 0.005*"state" + 0.005*"force" + 0.005*"innovation" + '
#   '0.005*"project" + 0.005*"develop" + 0.004*"work" + 0.004*"defense" + '
#   '0.004*"economic" + 0.004*"task" + 0.004*"economy" + 0.004*"issue" + '
#   '0.004*"society" + 0.004*"expert" + 0.004*"investment" + 0.004*"america" + '
#   '0.004*"recent"')]
