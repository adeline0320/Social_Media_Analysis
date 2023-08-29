import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
nltk.download('vader_lexicon')

import warnings
warnings.filterwarnings("ignore")


#################### Sentiment Analysis #######################

# Load dataset
col = ['id','text','created_at']
df_tweets = pd.read_csv("SmartiesUKI_dataset/SmartiesUKI_tweets.csv", header=None, names=col)

# remove duplicate entries
df_tweets_cleaned = df_tweets.drop_duplicates().copy() 
# change data type to date time
df_tweets_cleaned["created_at"] = pd.to_datetime(df_tweets_cleaned["created_at"])

### Function to clean the text of a tweet
def text_clean(text):
  text = text.lower() ### Convert to lowercase, to ensure consistency in the text data and avoids considering the same word as different.
  text = re.sub('\[.*?\]', '', text) ### Removes square bracket, to help in eliminate any irrelevant or non-textual content.
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text) ### Removes punctuation, to simplify the text and focuses on the essential words.
  text = re.sub('\w*\d\w*', '', text) ### Removes numbers, as it often do not contribute to the sentiment or meaningful analysis of the text.
  text = re.sub(r"\s+", " ", text) ### Remove extra whitespace, to ensure that words are properly separated and facilitates further text processing.
  text = re.sub(r"http\S+|www\S+|https\S+", "", text) ### Remove URLs, because URLs often appear in tweets but do not provide valuable information for sentiment analysis.
  text = re.sub(r"@\w+|#\w+", "", text) ### Remove mentions and hashtags, remove user mentions and hashtags as them not relevant in sentiment analysis or text processing.

  return text

### Apply the clean_text function to the 'text' column and store the cleaned text in a new column 'cleaned_text'
df_tweets_cleaned['cleaned_text'] = df_tweets_cleaned['text'].apply(text_clean)

### Function to perform sentiment analysis using VADER
def get_polarity_vader(text):
  return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

### Function to perform sentiment analysis using TextBlob
def get_polarity_textblob(text):
  return TextBlob(text).sentiment.polarity

### Apply sentiment analysis using VADER and store the polarity output in a new column 'VaderPolarity'
df_tweets_cleaned['VaderPolarity'] = df_tweets_cleaned['cleaned_text'].apply(get_polarity_vader)

### Apply sentiment analysis using TextBlob and store the polarity output in a new column 'TBPolarity'
df_tweets_cleaned['TBPolarity'] = df_tweets_cleaned['cleaned_text'].apply(get_polarity_textblob)

df_tweets_cleaned['VaderSentiment'] = ''
df_tweets_cleaned.loc[df_tweets_cleaned.VaderPolarity >= 0.05, 'VaderSentiment'] = 'POSITIVE'
df_tweets_cleaned.loc[df_tweets_cleaned.VaderPolarity.between(-0.05, 0.05, inclusive = 'left'), 'VaderSentiment'] = 'NEUTRAL'
df_tweets_cleaned.loc[df_tweets_cleaned.VaderPolarity < -0.05, 'VaderSentiment'] = 'NEGATIVE'

df_tweets_cleaned['TBSentiment'] = ''
df_tweets_cleaned.loc[df_tweets_cleaned.TBPolarity >= 0, 'TBSentiment'] = 'POSITIVE'
df_tweets_cleaned.loc[df_tweets_cleaned.TBPolarity == 0, 'TBSentiment'] = 'NEUTRAL'
df_tweets_cleaned.loc[df_tweets_cleaned.TBPolarity < 0, 'TBSentiment'] = 'NEGATIVE'

##############################################################



################## Response Time #######################
df = pd.read_json("SmartiesUKI_dataset/SmartiesUKI_replies_2.json")
reply = df[["text","username","timestamp"]]
reply["timestamp"] = pd.to_datetime(reply["timestamp"])

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

reply_list = []
smarties_list = []

for index, row in df.iterrows():
  if row['username'] == "@SmartiesUKI":
    smarties_list.append(df.loc[index])

time_list = [0,0,0,0,0,0]
time_range = ["0-60 minutes","1-2 hours","2-6 hours","6-12 hours","12-24 hours","24 hours+"]

for index, name in df.iterrows():
  for index2, reply in enumerate(smarties_list):
      if (len(reply["in_reply_to"])!=0):
        reply_user = reply["in_reply_to"][0]
        if(reply_user == name["username"]):
          create_tweet = name['timestamp']
          reply_tweet = reply['timestamp']
          time = abs(reply_tweet - create_tweet)
          num = timerange(time)
          time_list[num] +=1
          reply_list.append(time)
          break

df = pd.DataFrame(list(zip(time_range,time_list)),columns=["Time Range", "Number of Responses"])

total_minutes = sum(td.total_seconds()/60 for td in reply_list)
average_minutes= total_minutes / len(reply_list)

total_seconds = sum(td.total_seconds() for td in reply_list)
average_seconds = total_seconds / len(reply_list)
average_days = (total_seconds / (60 * 60 * 24))/len(reply_list)

########################################################



###################### Engagement #######################

df_engage = pd.read_json("SmartiesUKI_dataset/SmartiesUKI_engagement.json")
df_engage["timestamp"] = pd.to_datetime(df_engage["timestamp"])

df_engage['total_tweets'] = df_engage['total_tweets'].str.replace(',', '').astype(int)
df_engage['num_following'] = df_engage['num_following'].str.replace(',', '').astype(int)
df_engage['num_followers'] = df_engage['num_followers'].str.replace(',', '').astype(int)
df_engage['total_likes'] = df_engage['total_likes'].str.replace(',', '').astype(int)

df_engage['timestamp'] = pd.to_datetime(df_engage["timestamp"])
df_engage["Date"] = df_engage['timestamp'].dt.date

df_engage_sort = df_engage.sort_values(by='timestamp', ascending=False)
days = df_engage_sort['timestamp'].dt.date.unique()[:9]
df_engage_new = df_engage_sort[df_engage_sort['timestamp'].dt.date.isin(days)].copy()

# Calculate engagement rate as a percentage
df_engage_new['engagement_rate'] = (df_engage_new['likes'] + 
    df_engage_new['retweets'] + df_engage_new['replies'] + df_engage_new['quotes'])/(len(df_engage))

# Calculate the total engagement and total number of followers for each date
engagement_by_date = df_engage_new.groupby('Date').agg(
    {'engagement_rate': 'sum', 'num_followers': 'sum'}).reset_index()

# Calculate the average engagement rate per follower
engagement_by_date['avg_engagement_rate_per_follower'] = (engagement_by_date['engagement_rate'] / \
    4504)*100

#########################################################



###################### Reach #######################

df_hashtag = pd.read_json("SmartiesUKI_dataset/SmartiesUKI_hashtag.json")
df_hashtag["timestamp"] = pd.to_datetime(df_hashtag["timestamp"])

df_hashtag['total_tweets'] = df_hashtag['total_tweets'].str.replace(',', '').astype(int)
df_hashtag['num_following'] = df_hashtag['num_following'].str.replace(',', '').astype(int)
df_hashtag['num_followers'] = df_hashtag['num_followers'].str.replace(',', '').astype(int)
df_hashtag['total_likes'] = df_hashtag['total_likes'].str.replace(',', '').astype(int)
df_hashtag['timestamp'] = pd.to_datetime(df_hashtag["timestamp"])
df_hashtag["Date"] = df_hashtag['timestamp'].dt.date

df_hash_sort = df_hashtag.sort_values(by='timestamp', ascending=False)
days = df_hash_sort['timestamp'].dt.date.unique()[:30]
df_hashtag_new = df_hash_sort[df_hash_sort['timestamp'].dt.date.isin(days)].copy()

# Group the data by 'Date' and calculate the sum of 'num_followers' for each date
followers_by_date = df_hashtag_new.groupby('Date')['num_followers'].sum()

####################################################



###################### Impression #######################

days = df_hash_sort['timestamp'].dt.date.unique()[:30]
df_hashtag_new = df_hash_sort[df_hash_sort['timestamp'].dt.date.isin(days)].copy()

df_impress = df_hashtag_new.copy()

# Calculate impressions by multiplying followers with the number of tweets
df_impress['Impression'] = df_impress.groupby(['username', 'Date'])['num_followers'].transform('sum')

# Sum the impressions for each date
impressions_by_date = df_impress.groupby('Date')['Impression'].sum()

# Print the impressions of the hashtag for each date
total_impact = pd.DataFrame(impressions_by_date).reset_index()

#########################################################



###################### Dashboard #######################

st.title("SmartiesUKI Dashboard")

######## Response Time ########
st.markdown("<centering><h2>Response Time</h2></centering>", unsafe_allow_html=True)
col1_1, col1_2 = st.columns([1,3])

with st.container():
  with col1_1:
    st.markdown("#### Average Reply: ")
    # st.markdown(f"**Average seconds:** <br>{average_seconds:.0f}s", unsafe_allow_html=True)
    # st.markdown(f"**Average days:** <br>{average_days:.4f}", unsafe_allow_html=True)
    st.markdown(f"<ul><li><span style='color: red;font-size: 20px'><b>{average_seconds:.0f}</b></span> <span style='color: blue;font-size: 20px'>seconds</span></li></ul>", unsafe_allow_html=True)
    st.markdown(f"<ul><li><span style='color: red;font-size: 20px'><b>{average_days:.4f}</b></span> <span style='color: blue;font-size: 20px'>days</span></li></ul>", unsafe_allow_html=True)
    
  with col1_2:
    # Plotting the bar chart
    fig, ax = plt.subplots()
    ax.bar(df["Time Range"], df["Number of Responses"], color='blue')    
    ax.set_xlabel('Response Time Range')
    ax.set_ylabel('Number of Responses')
    ax.set_title('Smarties Response Time Distribution')
    ax.set_xticklabels(time_range, rotation=30)
    
    st.pyplot(fig)

######## Engagement, Sentiment analysis ########
col2_1, col2_2 = st.columns(2)
with st.container():
  with col2_1:
    ######## Engagement ########
    st.subheader("Engagement Rate")

    fig, ax = plt.subplots()

    ax.plot(np.array(engagement_by_date['Date']), 
            np.array(engagement_by_date['avg_engagement_rate_per_follower']))
    ax.set_xlabel('Date')
    ax.set_ylabel('Engagement')
    # ax.set_title('Total Smarties Engagement over Time')
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

  with col2_2:
    ######## Sentiment Analysis ########
    st.subheader("Sentiment Analysis")

    X = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
    X_axis = np.arange(len(X))

    fig, ax = plt.subplots()
    ax.bar(X_axis - 0.2, df_tweets_cleaned['VaderSentiment'].value_counts(), width = 0.4, label = 'VaderSentiment')
    ax.bar(X_axis + 0.2, df_tweets_cleaned['TBSentiment'].value_counts(), width = 0.4, label = 'TBSentiment')

    ax.set_xticks(X_axis)
    ax.set_xticklabels(X)
    ax.set_ylabel("Counts")
    # ax.set_title("Customer Sentiment")
    ax.legend()

    st.pyplot(fig)

######## Reach, Impression ########
st.markdown('<h5 style="text-align:center; background-color:#F08080; color:white;">Hashtag: <b>#SmartiesButtons</b></h5>', unsafe_allow_html=True)
col3_1, col3_2 = st.columns(2)
with st.container():
  with col3_1:
    ######## Reach ########
    st.subheader("Reach Rate")

    fig, ax = plt.subplots()
    ax.plot(np.array(followers_by_date.index),
            np.array(followers_by_date.values))    
    ax.set_xlabel('Date')
    ax.set_ylabel('Reach')
    # ax.set_title('Reach over Date')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.tick_params(axis='x', rotation=45)
    
    st.pyplot(fig)

  with col3_2:
    ######## Impression ########
    st.subheader("Impression Rate")
    fig, ax = plt.subplots()
    ax.plot(np.array(total_impact.Date),
            np.array(total_impact.Impression))    
    ax.set_xlabel('Date')
    ax.set_ylabel('Impression')
    # ax.set_title('Impression over Date')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.tick_params(axis='x', rotation=45)

    st.pyplot(fig)

#########################################################
