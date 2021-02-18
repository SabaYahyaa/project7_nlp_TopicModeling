
########################
#  import packages
########################
from newspaper import Article
from pprint import pprint
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
lemmatizer = WordNetLemmatizer()

#extract only the verb, noun, adjective
def get_wordnet_pos(word):
    #use post_tag, and extract the 1st
    tag=nltk.pos_tag([word])[0][1][0].upper()
    tag_dic={"J": wordnet.ADJ,   #extract adjective
             "N": wordnet.NOUN,  #extract noun
             "V":wordnet.VERB,   #extract ver
             #### TO DO, do not use adv
             "R":wordnet.ADV #extract adverb, I will remove this later
             }
    return (tag_dic.get(tag, wordnet.NOUN))

from nltk.corpus import stopwords #extend the stop words list
stop_words = stopwords.words('english')
##### extra words to be considered as stop words:
# 'most','much', 'yet', 'whole', 'may', 'could', 'th', 'okay', 'still', 'almost', 'every'
stop_words.extend(['U','non','actually,','sometimes,','sometimes','sometime,','sometime', 'would','really','great','lot'])

import string #get all the punctuation from string

exclude_string = set(string.punctuation)

import re
import pandas as pd

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
# ####################
# ###  Test Data
# ####################
# # I will test any thing I do on the following sentences, then I will apply on the text from url
# # I need to create a class, or function that does for all text later
# doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father ?."
# doc2 = "My father spends a lot of time driving my sister around to dance practice 324."
# doc3 = "actually, At BeCode, we like to learn. Sometime, we play games not win a price but to have fun!"
# doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
# doc5 = "      Health experts say that Sugar is not good for your lifestyle."
# text=[doc1, doc2, doc3]

###########################################
### 1. convert text to lower case
########################################
def CleanText(doc):
    # remove ", get only words
    word0=  re.sub(r'[^\w]', ' ', doc)
    #get words if the length is more than 1, remove word that has only one character
    mord_more_1=re.sub(r'\b\w{1,1}\b', '',  word0)
    #remove numbers,
    num_free = re.sub(r'\d+', '', mord_more_1)
    #convert the text to lower case, and the remove stopwords
    stop_free = " ".join([i for i in num_free.lower().split() if i not in stop_words])
    #remove punctuation
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude_string)
    #apply lemmatize, to get the abstract of a word
    #print([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in nltk.word_tokenize(sentence)])
    normalized = " ".join(lemmatizer .lemmatize(word, get_wordnet_pos(word)) for word in punc_free.split())
    #normalized = " ".join(lemmatizer.lemmatize(word) for word in punc_free.split())
    #apply stemmer (I will not use it, sometimes it cuts some letters from words, for that I do not understand them later
   # stemmerized= " ".join( stemmer.stem(word) for word in normalized.split() )
    return (normalized)

# cleaned_text=CleanText(doc5)
# print(cleaned_text)

if __name__=='__main__':

    #url= "https://artificialintelligence-news.com/2017/04/25/zingbox-aims-internet-trusted-things-bundles-ai-new-solution/"
    #url= "https://artificialintelligence-news.com/2017/04/25/ai-may-help-create-sustainable-data-centres-theres-work-still/"
    #url="https://artificialintelligence-news.com/2017/04/25/potential-trillion-dollar-b2b-bots-industry-decade-innovation-come/"
    url="https://artificialintelligence-news.com/2017/04/25/companies-investing-ai-today-expect-revenue-spike-2020/"
    article = Article(url)
    # #Call the download and parse methods to download information
    article.download()
    article.parse()

    #print(f"title={article.title}")
    #print(f"the text is= {article.text[:400]}")
    # print(f"length of this article=",{len(article.text)} )
    # #Runs nlp method to extract the keywords and summary of the article
    #article.nlp()
    # #Return article's keywords (tag)
    #print(f"the article keywords= {article.keywords}")
    #Return article's summary
    #print(f"the article summary= {article.summary}")

    pprint(article.text)
    print("----------------- after processing --------------------------")
    cleaned_text=CleanText(article.text)
    pprint(cleaned_text.split())







