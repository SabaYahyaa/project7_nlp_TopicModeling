# project7_nlp_TopicModeling
nlp_Topic Modeling is a NLP application that uses LDA-MALLET (Latent Dirichlet allocation, Topic Modeling) and Xgboost to classify the topics of newspapers.

## Features:
Dealing with news_data.json under the data folder.
1. Preprocessed each text (document) using preprocessed_text.py (text cleaning)
2. Create X (features) using Creating_X_TFIDF.py (applying TFIDF Vectorizing on document to create numerical features)
3. Create y (label) for each document using LDA-MALLET. Each y is the dominated topic.
4. Split the data (X and y) and train Xgboost.
5. Use Xgboost to find the label (dominated topic) for each document.

## highlights:

We create newspaper app, each time you select some topics from a checkbox topics, you get all the available urls. These urls are sent to your email.

