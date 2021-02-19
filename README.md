## Team member names:
1. Saba Yahyaa
2. Mikael Dominguez
3. Adam Flasse

# project7_nlp_TopicModeling
nlp_Topic Modeling is a NLP application that uses LDA-MALLET (Latent Dirichlet allocation, Topic Modeling) and Xgboost to classify the topics of newspapers.

## Features:
Dealing with news_data.json under the data folder.
1. Preprocessed each text (document) using preprocessed_text.py (text cleaning)
2. Create X (features) using Creating_X_TFIDF.py (applying TFIDF Vectorizing on document to create numerical features)
3. Create y (label) for each document using LDA-MALLET. Each y is the dominated topic.
4. Split the data (X and y) and train Xgboost.
5. Use Xgboost to find the label (dominated topic) for a new document. 
6. Use LDA-MALLET to find the topics for a new document.

We specified the following 17 topics:
0. AI in fake news  (Social Media Marketing)
1. AI in Human Discrimination
2. AI in Material and Energy research
3. AI in Voice Assistant
4. AI in Business
5. AI in AI Autonomous vehicle
6. AI in Hadware (Chip)
7. AI in Human question/problem answer
8. AI in Image (NN)
9. AI in Arts
10. AI in Robots
11. AI in Medecine
12. AI in Legal Service
13. AI in Industry (sillicon valley)
14. AI in Academic Research
15. AI in Game
16. AI in DNN and Machine learning application


## highlights:

We create  [https://ainewspaper.herokuapp.com/] (# newslAItter) app. Each time you select some topics from available AI topics, you enter you email address.
All the selected topics will be sent to your email.

