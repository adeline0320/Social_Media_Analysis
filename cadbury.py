import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
import re
import numpy as np


nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Set the title 
st.markdown(""" <style> 
@import url('https://fonts.googleapis.com/css?family=Pacifico');

.font {
    font-size:80px ; 
    font-family: 'Pacifico', serif;
    color: #482683;
    text-align: center;
    transform: translateY(-10%);
} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Cadbury UK Dashboard</p>', unsafe_allow_html=True)

##################  Response Time #######################
#Function to return range of time
def timerange(time):
   min = time.total_seconds()/60
   if (min <= 60):
      return 0
   elif (min > 60 and min <= 120):
      return 1
   elif (min > 120 and min <= 360):
      return 2
   elif (min > 360 and min <= 720):
      return 3
   elif (min > 720 and min <= 1440):
      return 4
   else:
      return 5

data = pd.read_json("CadburyUK_Dataset/Dataset_for_replies.json")
response = pd.DataFrame(data)

#take only relevant column
res_df = response[["tweet_id","text","username","timestamp","in_reply_to"]]

#change to datetime format
res_df['date'] = pd.to_datetime(res_df['timestamp'])


reply_list = []
cadbury_reply =[]
#Take only replies from cadbury
for index, row in res_df.iterrows():
  if row['username'] == "@CadburyUK":
    cadbury_reply.append(res_df.loc[index])

time_list = [0,0,0,0,0,0]
time_range = ["0-60 minutes","1-2 hours","2-6 hours","6-12 hours","12-24 hours","24 hours+"]

#iterate the dataset
for index,name in res_df.iterrows():
  for index2,reply in enumerate(cadbury_reply):
      #If there's reply to someone, get the reply username
      if (len(reply["in_reply_to"])!=0):
        reply_user = reply["in_reply_to"][0]
        #if the username is same as the reply username
        if(reply_user == name["username"]):
          #find the tweet created date and the date of Cadbury reply the tweet
          create_tweet = name['timestamp']
          reply_tweet = reply['timestamp']
          #find the time difference
          time = abs(reply_tweet - create_tweet)
          #check what is the range of the time difference and return the output
          num = timerange(time)
          #add the time range in the array
          time_list[num] +=1
          #append the time to be used at the next step
          reply_list.append(time)
          break

#put the time range list into a dataframe to be used to show bar chart
df = pd.DataFrame(list(zip(time_range,time_list)),columns=["Time Range", "Number of Responses"])

#find the average seconds and days for cadbury to reply
total_seconds = sum(td.total_seconds() for td in reply_list)
average_seconds = total_seconds / len(reply_list)
average_days = (total_seconds / (60 * 60 * 24))/len(reply_list)


#Plotting the bar chart
fig_time = plt.subplots()
####################   Sentiment Analysis ###################
#In this section, the replies data is used and is filtered to only take tweets within 7 days.

sid = SentimentIntensityAnalyzer()

# Get the maximum date in the dataframe
max_date = res_df['date'].max()

# Calculate the start date by subtracting 7 days from the maximum date
start_date = max_date - timedelta(days=7)
filtered_data =  res_df[(res_df['date'] >= start_date) & (res_df['date'] <= max_date)]

res_7_days = pd.DataFrame(filtered_data)
df_sa = res_7_days.copy()

#Data Cleaning

def clean(text):
    #remove url,rt,symbols,hashtag
    text = re.sub(r'http\S+',' ', text)

    # Remove usernames (mentions)
    text= re.sub(r'@\w+', '',  text)

    # Remove hashtags
    text = re.sub(r'#\w+', '',  text)

    # Remove 'RT' (retweet indicator)
    text = re.sub(r'\bRT\b', '', text)

    # Remove special characters and punctuations
    text = re.sub(r'[^\w\s]', '',  text)

     # Convert to lowercase
    text = text.lower()

    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ',  text).strip()
    return text

# add the cleaned data to the new column
df_sa['cleaned_text'] = df_sa['text'].apply(clean)

#remove cadbury reply
df_sa = df_sa.drop(df_sa[df_sa['username'] == '@CadburyUK'].index)

#function to perform sentiment analysis using VADER
def get_polarity_vader(text):
  return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

#function to perform sentiment analysis using TextBlob
def get_polarity_textblob(text):
  return TextBlob(text).sentiment.polarity

#apply sentiment analysis using VADER and store the polarity output in a new column 'VaderPolarity'
df_sa['VaderPolarity'] = df_sa['cleaned_text'].apply(get_polarity_vader)

#apply sentiment analysis using TextBlob and store the polarity output in a new column 'TBPolarity'
df_sa['TBPolarity'] =df_sa['cleaned_text'].apply(get_polarity_textblob)

#get the sentiment of each VADER score
df_sa['VaderSentiment'] = ''
df_sa.loc[df_sa.VaderPolarity >= 0.05, 'VaderSentiment'] = 'POSITIVE'
df_sa.loc[df_sa.VaderPolarity.between(-0.05, 0.05, inclusive = 'left'), 'VaderSentiment'] = 'NEUTRAL'
df_sa.loc[df_sa.VaderPolarity < -0.05, 'VaderSentiment'] = 'NEGATIVE'

#get the sentiment of each TB score
df_sa['TBSentiment'] = ''
df_sa.loc[df_sa.TBPolarity >= 0, 'TBSentiment'] = 'POSITIVE'
df_sa.loc[df_sa.TBPolarity == 0, 'TBSentiment'] = 'NEUTRAL'
df_sa.loc[df_sa.TBPolarity < 0, 'TBSentiment'] = 'NEGATIVE'

#count each sentiment and put into dictionary to used in plotting bar chart
vader_sentiment = df_sa["VaderSentiment"].value_counts().to_dict()
tb_sentiment =  df_sa["TBSentiment"].value_counts().to_dict()

#********************Hashtag Data Collection and cleaning *************

hashtag_df = pd.read_csv('CadburyUK_Dataset/hashtag_tweets.csv')
hashtag_df = hashtag_df.drop(["Unnamed: 0"],axis = 1)
hashtag_df['hour'] = pd.to_datetime(hashtag_df['createdate']).dt.hour
hashtag_df['date'] = pd.to_datetime(hashtag_df['createdate']).dt.date
hashtag_df['period'] = (hashtag_df['hour']% 24 + 8) // 4
hashtag_df['period'].replace({1: '03',
                      2: '06',
                      3: '09',
                      4: '12',
                      5: '15',
                      6: '18',
                      7: '21'}, inplace=True)
#*********************** Reach Rate ********************
reach_df = hashtag_df.copy()

# Drop duplicate username
reach_df = reach_df.drop_duplicates(subset=['username'])

# Calculate reach by summing the number of followers
reach_df['reach'] = reach_df.groupby(['date', 'period'])['followers'].transform('sum')

# Sum the reach for each date and period
reach_by_date_period = reach_df.groupby(['date', 'period'])['reach'].sum()

# Put the output into dataframe
reach_df = pd.DataFrame(reach_by_date_period).reset_index()

# Sort the DataFrame by date 
min_value = reach_df['reach'].min()
max_value = reach_df['reach'].max()
reach_df['date_period'] = reach_df['date'].astype(str) + ' ' + reach_df['period']

###########################################  Impression ##########################################################
impression =  hashtag_df.copy()

# Get the date from the 'createdate' variable
impression['date'] = pd.to_datetime(impression['createdate']).dt.date

# Calculate impressions by multiplying followers with the number of tweets
impression['impressions'] = impression.groupby(['date', 'username'])['followers'].transform('sum')

# Sum the impressions for each date
impressions_by_date = impression.groupby('date')['impressions'].sum()

# Put the output into dataframe
impressions_df = pd.DataFrame(impressions_by_date).reset_index()


# impressions_df ['date'] = pd.to_datetime(impressions_df ['date'])  

########################################## Engagement Rate ####################################################
#Engagement rate
enga = pd.read_json("CadburyUK_Dataset/Engagement.json")
engagement = enga.copy()

#Select relvevant columns
engagement = engagement[["created_at","reply_count","retweet_count","favorite_count","quote_count"]]

#Get date of the tweet
engagement["created_at"] = pd.to_datetime(engagement["created_at"])
engagement["date"] = engagement["created_at"].dt.date

#set number of followers and number of tweets available in Cadbury timeline 
followers = 3131000
num_tweets = 14

#Calculate the engagement rate for each tweet
engagement["engagement_rate"] = (((engagement.reply_count + engagement.retweet_count + engagement.favorite_count + engagement.quote_count) / num_tweets) / followers) * 100

#Sort the output based on date
engagement = engagement.sort_values('date')

# Create the dot line chart using Matplotlib
eng_fig = plt.subplots()


##################### Layout Application ##################
cont1 = st.container()
col1, col2= st.columns((1,1))

with cont1:
    st.markdown(""" <style> 

        .res_time {
        font-size:30px ; 
        color: #9C7E46;
        text-align: center;
        transform: translateY(-50%);
        font-family:monospace , sans-serif
        } 
        
        </style> """, unsafe_allow_html=True)
    st.markdown(""" <style> 

        .res_time_display {
        font-size:20px ; 
        color: black;
        text-align: center;
        transform: translateY(-50%);
        } 
        
        </style> """, unsafe_allow_html=True)
   
    with col1:
        st.markdown('<p class="res_time"><strong>Response Time</strong></p>', unsafe_allow_html=True)
        second = round(average_seconds,2)
        days = round(average_days,2)
       
        

        st.markdown('<p class="res_time_display">Average Response time in :</p>', unsafe_allow_html=True)
        
        st.markdown(f"<ul style='text-align: center; list-style-type: none; padding-left: 0;font-size : 100px;'><li>&#8226; <strong style='color: red;font-size: 20px'>{second}</strong> <span style='color: blue;font-size: 20px'>seconds</span></li></ul>", unsafe_allow_html=True)
        st.markdown(f"<ul style='text-align: center; list-style-type: none; padding-left: 0;'><li>&#8226; <strong style='color: red;font-size: 20px'>{days}</strong> <span style='color: blue;font-size: 20px'>days</span></li></ul>", unsafe_allow_html=True)




        trace = go.Bar(
            x = df["Time Range"],
            y = df["Number of Responses"],
            text=df["Number of Responses"],
            textposition='auto'
        )

        data = [trace]
        fig_time= go.Figure(data = data)
        fig_time.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
           paper_bgcolor="white",
            height = 300)
        
        fig_time.update_xaxes(title ="Time",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
        fig_time.update_yaxes(title ="Frequency",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
       
        # Render the plotly graph with a fixed width using CSS
        st.plotly_chart(fig_time, use_container_width=True, config={'displayModeBar': False})
   
    with col2:
        st.markdown('<p class="res_time"><strong>Sentiment Analysis</strong></p>', unsafe_allow_html=True)
        # Create a figure and axis
        x_data = list(vader_sentiment.keys())
        y_data1 = list(vader_sentiment.values())
        y_data2 = list(tb_sentiment.values())

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data1,
            text=y_data1, 
            name="Vader Sentiment"
        ))

        fig.add_trace(go.Bar(
            x=x_data,
            y=y_data2,
            text=y_data2, 
            name="TextBlob Sentiment"
        ))
        fig.update_layout(
            barmode = 'group',
            autosize=True,
            width=400, height=450,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor="white")
        
        fig.update_xaxes(title ="Category",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
        fig.update_yaxes(title ="Frequency",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
        
        st.plotly_chart(fig, use_container_width=True,config={'displayModeBar': False})

st.markdown('#')
cont2 = st.container()
with cont2:
    st.markdown('<p class="res_time"><strong>Engagement Rate</strong></p>', unsafe_allow_html=True)
    fig = px.line(
        x = engagement["date"],
        y = engagement["engagement_rate"]
    )

    fig.update_layout(
        autosize=True,
        width=400, height=300,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white")
    
    fig.update_xaxes(title = "Months",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
    fig.update_yaxes(title ="Frequency", title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
   
    st.plotly_chart(fig, use_container_width=True,config={'displayModeBar': False})
st.markdown("#")
tag = st.container()

with tag:
   st.markdown(
      """
    <div style='border: 1px solid black; padding: 10px; background-color: #CBB386; text-align: center;'>
        <span style='color: #482683;font-size: 20px ;font-family:monospace , sans-serif;'>
            Hashtag # : #Don’tSearchTwirlMint
        </span>
    </div>
    """,unsafe_allow_html=True)

st.markdown('##')
    
container2 = st.container()
col4, col5= st.columns((2,2))
with container2:
     
    with col4:

        st.markdown('<p class="res_time"><strong>Reach Rat</strong>e</p>', unsafe_allow_html=True)
        fig_reach = px.line(
            x = reach_df["date_period"],
            y = reach_df["reach"],
            height=400
        )

        fig_reach.update_layout(
            width=10,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white")

        fig_reach.update_xaxes(title ="Months",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
        fig_reach.update_yaxes(title ="Reach Rate",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
    
        st.plotly_chart(fig_reach, use_container_width=True,config={'displayModeBar': False})


    with col5:

        st.markdown('<p class="res_time"><strong>Impression Rate</strong></p>', unsafe_allow_html=True)
        fig_imp = px.line(
            x = impressions_df["date"],
            y = impressions_df["impressions"],
        )

        fig_imp.update_layout(
            autosize=True,
            width=100,  # Set the desired width
            height=400,  # Set the desired height
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="white")

        fig_imp.update_xaxes(title ="Months",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
        fig_imp.update_yaxes(title ="Impression Rate",title_font_color="black",tickfont=dict(family='Rockwell', color='black', size=14))
    
        st.plotly_chart(fig_imp,config={'displayModeBar': False},use_container_width=True)
