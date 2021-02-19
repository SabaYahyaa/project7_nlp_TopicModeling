
"""
Apply LDA-MALLET on Unseen Data
"""

"""
I will apply Xgboost of a given url
"""
from preprocess_text import CleanText #for preprocessed text
from Creating_X_TFIDF import tfidf_vec #for creating numerical data
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
import joblib
import numpy as np





def apply_model_LdaMallet(lst_text):
    dominated_topic=None
    map_dic_topics = {
        0: "AI fake news  (Social Media Marketing)",
        1: "Human Discrimination",
        2: "Material and Energy research",
        3: "Voice Assistant",
        4: "AI Business",
        5: "AI Autonomous vehicle",
        6: "AI Hadware (Chip)",
        7: "AI Human question/problem answer",
        8: "AI Image (NN)",
        9: "AI Arts",
        10: "AI Robots",
        11: "AI Medecine",
        12: "AI Legal Service",
        13: "AI Industry (sillicon valley)",
        14: "AI Academic Research",
        15: "AI Game",
        16: "AI DNN and Machine learning application    (Development and Application"
    }
    #create preprocessed text
    all_data=list()
    for t in lst_text:
        cleaned_text=None
        try:
            cleaned_text = CleanText(t).split() #create a list of token
        except:
            cleaned_text=None
        all_data.append(cleaned_text)

    # #load a model

    unseen_data=all_data
    print(unseen_data)
    print(type(unseen_data))

    # #load a model
    lda_mallet = gensim.models.wrappers.LdaMallet.load('./data/ldamallet_17.gensim')
    # with open('./data/all_data.json') as all_data:
    #     all_data=json.load(all_data)
    # unseen_data=[all_data[0]]
    # print(unseen_data)
    # print(type(unseen_data))
    # # create a dictionary of individual words and filter the dictionary
    dictionary_new = gensim.corpora.Dictionary(unseen_data)

    # # define the bow_corpus
    bow_corpus_new = [dictionary_new.doc2bow(doc) for doc in np.array(unseen_data)]

    all_topics = lda_mallet[bow_corpus_new[:len(bow_corpus_new)]]
    dominated_topics=[all_topics[i][0][0] for i in range(0,len(all_topics))]
    #print(dominated_topics)

    topics_name=list()
    for t in dominated_topics:
        topics_name.append(map_dic_topics.get(int(t)))

    return (topics_name)

if __name__=="__main__":
    # the input is a list of text that is extracted from url
    lst_text=[
        "Enterprise data centre provider Aegis Data argues in its latest note that utilising artificial intelligence (AI) could be key in winning the battle for sustainable data centres.\n\n“There’s no escaping the reality that as more connected devices and technology trends sweep the market, more demands will be placed on the data centre to provide the high-powered servers and cooling systems required,” said Greg McCulloch, CEO of Aegis. “But in the pursuit of guaranteeing performance, it is having an accumulative effect on the global share of data centre emissions… it’s not an understatement that the industry needs to take immediate action and AI may just be the solution in order to help achieve more sustainable results.”\n\nFor data centre providers, PUE (power usage effectiveness) figures have traditionally been the name of the game for sustainability, dividing energy for the whole facility by the energy for the facility’s IT equipment and aiming for an ideal score of 1.0.\n\nBack in April 2015 this reporter attended the opening of a new Rackspace data centre (below) in West Crawley, which had an impressive 1.15 PUE rating; the average was nearer 1.7 with Rackspace admitting themselves their target was 1.25. The data centre utilised other natural benefits, including cooling using natural air – a feature that was tailor made for the UK – while the design was donated to the Open Compute Project.\n\nIt was pretty impressive stuff all round. Yet not everyone agrees that PUE is the way forward. Professor Ian Bitterlin, chair of the British Computer Society’s data centre group on server lifecycle, told E&T last year that “improving server effectiveness is the only way to improve data centre effectiveness”, and that people are erroneously putting PUE as a data centre ‘goodness’ metric.\n\nAs far as Aegis is concerned, the work Google is undertaking by putting its DeepMind technology in its data centres – and experiencing a 15% energy efficiency improvement in the process – is on the right lines, but not without its flaws. “[AI is] a technology that’s very much in its infancy, and if it is to overtake human interaction in the data centre, then it must be rigorously researched and tested to guarantee performance,” said McCulloch. “But once this hurdle is overcome, AI has the ability to provide a comprehensive visualisation, automation and monitoring process that can envision the necessary power, cooling and energy requirements needed.”\n\nGoogle is not the only company to be experimenting with cognitive intelligence in its data centre facilities; at the end of last year IBM announced the launch of four new centres in the UK, and revealed how one customer, travel operator Thomson, was trialling new search functionality based on supercomputer Watson.\n\n“With the wide range of operations that occur in a data centre, the ability to eliminate human error and have an ‘always on’ approach will go a long way in helping reduce energy consumption,” added McCulloch.\n\nInterested in hearing industry leaders discuss subjects like this and sharing their use-cases? Attend the co-located AI & Big Data Expo events with upcoming shows in Silicon Valley, London and Amsterdam to learn more. Co-located with the IoT Tech Expo, Blockchain Expo and Cyber Security & Cloud Expo so you can explore the future of enterprise technology in one place.",
        "From Domino’s Pizza, to Uber, to Bank of America, bots are one of the hottest properties in business tech right now and opening up new avenues. Yet according to Beerud Sheth, CEO of chatbot platform provider Gupshup: “I see a decade’s worth of innovation yet to come.”\n\nAt Mobile World Congress, Sheth was fielding multiple queries as your reporter sidled up for a chat; something of a revelation given the location was the traditionally relatively sparsely populated Hall 8.1. Yet this further validates the Gupshup chief’s vision. “I do believe it’s the next big thing in tech,” says Sheth. “It’s a big paradigm shift that’s going to impact virtually every business.”\n\nOne business in particular with whom this has already made an impact is VentureBeat, who enlisted Gupshup to build a voice bot which reads the latest news headlines to users. Another, announced recently, is India’s Yes Bank. Sheth says that ‘it’s great to get validation from real customers at credible businesses’ – the company is talking to retail and media organisations among others – while noting that with sensitive data at hand, particularly on the banking side, bots may not be all-encompassing.\n\n“Every channel has its strengths and weaknesses,” he explains. “There are data concerns with SMS, there are data concerns with voice, there are data concerns with apps and with websites. What you do is they figure out there’s certain information you ca deliver in certain channels and information you don’t – but that doesn’t rise to the level of ‘should we use it or not?’\n\n“Getting balances and so on may be okay [on its own], but if you actually want to move money out, then maybe there are two or more checks, and they do two factor or sometimes three factor authentication across different channels.\n\n“It’s not like somebody can break the bank, so to speak,” adds Sheth. “There are different levels of security for different things you’re doing, and there are ways to address all of them.”\n\nGiven it’s still early days, the journey element of potential customer implementation is vital. Sheth admits his primary role at MWC is to ‘evangelise and increase awareness of bots’ – in other words, what they can do and why they are required – but after that, the opportunity is almost limitless. As he explains – and the missive on the company’s website is pretty clear – bots will be the next big breakthrough, eventually superseding the smartphone.\n\n“I think by next year, you’re going to see some breakout bots, breakout use cases, real success stories, bots that are engaging, bots that are monetising and so on, in a two to three year timeframe you’re probably going to see millions of bots, and in a decade, who knows?”\n\nTo learn more about Bot & Virtual Assistant Development register for your free pass to the AI Expo North America conference (29-30th November 2017) in Santa Clara, CA today!\n\nInterested in hearing industry leaders discuss subjects like this and sharing their use-cases? Attend the co-located AI & Big Data Expo events with upcoming shows in Silicon Valley, London and Amsterdam to learn more. Co-located with the IoT Tech Expo, Blockchain Expo and Cyber Security & Cloud Expo so you can explore the future of enterprise technology in one place.",
        "Enterprise data centre provider Aegis Data argues in its latest note that utilising artificial intelligence (AI) could be key in winning the battle for sustainable data centres.\n\n“There’s no escaping the reality that as more connected devices and technology trends sweep the market, more demands will be placed on the data centre to provide the high-powered servers and cooling systems required,” said Greg McCulloch, CEO of Aegis. “But in the pursuit of guaranteeing performance, it is having an accumulative effect on the global share of data centre emissions… it’s not an understatement that the industry needs to take immediate action and AI may just be the solution in order to help achieve more sustainable results.”\n\nFor data centre providers, PUE (power usage effectiveness) figures have traditionally been the name of the game for sustainability, dividing energy for the whole facility by the energy for the facility’s IT equipment and aiming for an ideal score of 1.0.\n\nBack in April 2015 this reporter attended the opening of a new Rackspace data centre (below) in West Crawley, which had an impressive 1.15 PUE rating; the average was nearer 1.7 with Rackspace admitting themselves their target was 1.25. The data centre utilised other natural benefits, including cooling using natural air – a feature that was tailor made for the UK – while the design was donated to the Open Compute Project.\n\nIt was pretty impressive stuff all round. Yet not everyone agrees that PUE is the way forward. Professor Ian Bitterlin, chair of the British Computer Society’s data centre group on server lifecycle, told E&T last year that “improving server effectiveness is the only way to improve data centre effectiveness”, and that people are erroneously putting PUE as a data centre ‘goodness’ metric.\n\nAs far as Aegis is concerned, the work Google is undertaking by putting its DeepMind technology in its data centres – and experiencing a 15% energy efficiency improvement in the process – is on the right lines, but not without its flaws. “[AI is] a technology that’s very much in its infancy, and if it is to overtake human interaction in the data centre, then it must be rigorously researched and tested to guarantee performance,” said McCulloch. “But once this hurdle is overcome, AI has the ability to provide a comprehensive visualisation, automation and monitoring process that can envision the necessary power, cooling and energy requirements needed.”\n\nGoogle is not the only company to be experimenting with cognitive intelligence in its data centre facilities; at the end of last year IBM announced the launch of four new centres in the UK, and revealed how one customer, travel operator Thomson, was trialling new search functionality based on supercomputer Watson.\n\n“With the wide range of operations that occur in a data centre, the ability to eliminate human error and have an ‘always on’ approach will go a long way in helping reduce energy consumption,” added McCulloch.\n\nInterested in hearing industry leaders discuss subjects like this and sharing their use-cases? Attend the co-located AI & Big Data Expo events with upcoming shows in Silicon Valley, London and Amsterdam to learn more. Co-located with the IoT Tech Expo, Blockchain Expo and Cyber Security & Cloud Expo so you can explore the future of enterprise technology in one place.",
        "From Domino’s Pizza, to Uber, to Bank of America, bots are one of the hottest properties in business tech right now and opening up new avenues. Yet according to Beerud Sheth, CEO of chatbot platform provider Gupshup: “I see a decade’s worth of innovation yet to come.”\n\nAt Mobile World Congress, Sheth was fielding multiple queries as your reporter sidled up for a chat; something of a revelation given the location was the traditionally relatively sparsely populated Hall 8.1. Yet this further validates the Gupshup chief’s vision. “I do believe it’s the next big thing in tech,” says Sheth. “It’s a big paradigm shift that’s going to impact virtually every business.”\n\nOne business in particular with whom this has already made an impact is VentureBeat, who enlisted Gupshup to build a voice bot which reads the latest news headlines to users. Another, announced recently, is India’s Yes Bank. Sheth says that ‘it’s great to get validation from real customers at credible businesses’ – the company is talking to retail and media organisations among others – while noting that with sensitive data at hand, particularly on the banking side, bots may not be all-encompassing.\n\n“Every channel has its strengths and weaknesses,” he explains. “There are data concerns with SMS, there are data concerns with voice, there are data concerns with apps and with websites. What you do is they figure out there’s certain information you ca deliver in certain channels and information you don’t – but that doesn’t rise to the level of ‘should we use it or not?’\n\n“Getting balances and so on may be okay [on its own], but if you actually want to move money out, then maybe there are two or more checks, and they do two factor or sometimes three factor authentication across different channels.\n\n“It’s not like somebody can break the bank, so to speak,” adds Sheth. “There are different levels of security for different things you’re doing, and there are ways to address all of them.”\n\nGiven it’s still early days, the journey element of potential customer implementation is vital. Sheth admits his primary role at MWC is to ‘evangelise and increase awareness of bots’ – in other words, what they can do and why they are required – but after that, the opportunity is almost limitless. As he explains – and the missive on the company’s website is pretty clear – bots will be the next big breakthrough, eventually superseding the smartphone.\n\n“I think by next year, you’re going to see some breakout bots, breakout use cases, real success stories, bots that are engaging, bots that are monetising and so on, in a two to three year timeframe you’re probably going to see millions of bots, and in a decade, who knows?”\n\nTo learn more about Bot & Virtual Assistant Development register for your free pass to the AI Expo North America conference (29-30th November 2017) in Santa Clara, CA today!\n\nInterested in hearing industry leaders discuss subjects like this and sharing their use-cases? Attend the co-located AI & Big Data Expo events with upcoming shows in Silicon Valley, London and Amsterdam to learn more. Co-located with the IoT Tech Expo, Blockchain Expo and Cyber Security & Cloud Expo so you can explore the future of enterprise technology in one place."
    ]

    topics=apply_model_LdaMallet(lst_text)
    print(topics)

