from flask import Flask, render_template, flash, request, redirect, url_for, jsonify
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import os
#from preprocess_text import CleanText
#from newspaper import Article
import json
import itertools
#importing the smtplib library
import smtplib
from email.message import EmailMessage

EMAIL_ADDRESS = 'newslaitters@gmail.com'
EMAIL_PASSWORD = 'adamsabamika'


app=Flask(__name__)

#create the list label
label_list = [  "Social Media (Fake News)",
                "Human Discrimination",
                "Material and Energy",
                "Voice Assistant",
                "Business",
                "Autonomous Vehicle",
                "Hadware (Chip)",
                "Answer human questions and problems",
                "Image Recognition by DNN",
                "Art",
                "Robots",
                "Medical",
                "Legal Service (Government)",
                "Industry (Sillicon Aalley)",
                "Academic Research",
                "Game",
                "Development and Application" ]

@app.route('/alive', methods=["GET"])
def index():
    if request.method == "GET":
        return "Alive"
    else:
        return "Server not working"


@app.route('/', methods=["GET", "POST"])
def urls():
    if request.method == 'POST':

        selected_topics=request.form.getlist('check')
        to_email = request.form.get('email')
        topics=[int(t) for t in selected_topics]
        print(topics)
        #extract the csv file that contains
        df_data=pd.read_csv('./data/news_labeled_data.csv')
        #create a list that contains all the mentioned topics
        url_selected_topics=list()
        newsletters = {}
        for t in topics:
            #filtte the df accordin to that topic
            to_be_filter_topic = df_data['Dominant_Topic'].apply(lambda x: x == t)
            filtered_url = df_data.loc[to_be_filter_topic, 'url'].values
            url_selected_topics.append(filtered_url)
            newsletters[label_list[t]]= filtered_url[:5]

        html_newsletters = (render_template('minty_v2.html', newsletters=newsletters))
        try:
            msg = EmailMessage()
            msg['Subject'] = 'Your newslAIter about AI'
            msg['From'] = EMAIL_ADDRESS 
            msg['To'] = to_email
            msg.set_content(html_newsletters, subtype='html')


            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD) 
                smtp.send_message(msg)
            return html_newsletters
        except:
            print("Error, email was not sent")
            return render_template('index.html')
    
    
    return render_template('index.html')









if __name__=="__main__":

    #### this is a copy from our track
    #app.run(debug=True)
    # You want to put the value of the env variable PORT if it exist (some services only open specifiques ports)
    port = int(os.environ.get('PORT', 5000))
    # Threaded option to enable multiple instances for
    # multiple user access support
    # You will also define the host to "0.0.0.0" because localhost will only be reachable from inside de server.
    app.run(host="0.0.0.0", threaded=True, port=port)
    #app.run(debug=True, host='0.0.0.0', port=port)

