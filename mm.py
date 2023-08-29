################# Load libraries ################################
import warnings
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

warnings.filterwarnings("ignore")
#################################################################


######################## Load Datasets ##########################
DF_RES = pd.read_json("MM_dataset/M&M_replies_2.json")
DF_RES["created_at"] = pd.to_datetime(DF_RES["created_at"])

col = ['id', 'text', 'created_at']
DF_SENT = pd.read_csv(
    "MM_dataset/M&M_tweets.csv", header=None, names=col)

# st.write(DF_SENT)
DF_CLEAN = DF_SENT.drop_duplicates().copy()  # remove duplicate entries
DF_CLEAN["created_at"] = pd.to_datetime(
    DF_CLEAN["created_at"])  # change data type to date time

DF_HASH = pd.read_json("MM_dataset/M&M_hashtag.json")
DF_HASH["timestamp"] = pd.to_datetime(DF_HASH["timestamp"])

DF_ENG = pd.read_json("MM_dataset/M&M_engagement.json")
DF_ENG["timestamp"] = pd.to_datetime(DF_ENG["timestamp"])
###################################################################

######################### Response ################################
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
mms_list = []

for index, row in DF_RES.iterrows():
  # if row['user_name'] == DF_RES['user_name'][0]:
    if row['user_name'] == "M&M’S":
        mms_list.append(DF_RES.loc[index])

# st.write(DF_RES['user_name'][0])
# st.write(mms_list)

time_list = [0,0,0,0,0,0]
time_range = ["0-60 minutes","1-2 hours","2-6 hours","6-12 hours","12-24 hours","24 hours+"]

for index, name in DF_RES.iterrows():
  for index2, reply in enumerate(mms_list):
      if (len(reply["user_mentions"])!=0):
        reply_user = reply["user_mentions"][0]
        if(reply_user == name["user_name"]):
            create_tweet = name['created_at']
            reply_tweet = reply['created_at']
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
####################################################################

################### Calculation of Reach Rate ######################
DF_HASH['total_tweets'] = DF_HASH['total_tweets'].str.replace(
    ',', '').astype(int)
DF_HASH['num_following'] = DF_HASH['num_following'].str.replace(
    ',', '').astype(int)
DF_HASH['num_followers'] = DF_HASH['num_followers'].str.replace(
    ',', '').astype(int)
DF_HASH['total_likes'] = DF_HASH['total_likes'].str.replace(
    ',', '').astype(int)
DF_HASH['timestamp'] = pd.to_datetime(DF_HASH["timestamp"])
DF_HASH["Date"] = DF_HASH['timestamp'].dt.date
DF_SORTED = DF_HASH.sort_values(by='timestamp', ascending=False)
DAYS_30 = DF_SORTED['timestamp'].dt.date.unique()[:30]
DF_HASH_NEW = DF_SORTED[DF_SORTED['timestamp'].dt.date.isin(DAYS_30)].copy()
DF_REACH = DF_HASH_NEW

# Group the data by 'Date' and calculate the sum of 'num_followers' for each date
followers_by_date = DF_REACH.groupby('Date')['num_followers'].sum()

followers_df = pd.DataFrame(
    {'Date': followers_by_date.index, 'TotalFollowers': followers_by_date.values})

#####################################################################


################# Calculation of Sentiment Analysis #################
# Function to clean the text of a tweet
def text_clean(text):
    # Convert to lowercase, to ensure consistency in the text data and avoids considering the same word as different.
    text = text.lower()
    # Removes square bracket, to help in eliminate any irrelevant or non-textual content.
    text = re.sub('\[.*?\]', '', text)
    # Removes punctuation, to simplify the text and focuses on the essential words.
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    # Removes numbers, as it often do not contribute to the sentiment or meaningful analysis of the text.
    text = re.sub('\w*\d\w*', '', text)
    # Remove extra whitespace, to ensure that words are properly separated and facilitates further text processing.
    text = re.sub(r"\s+", " ", text)
    # Remove URLs, because URLs often appear in tweets but do not provide valuable information for sentiment analysis.
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove mentions and hashtags, remove user mentions and hashtags as them not relevant in sentiment analysis or text processing.
    text = re.sub(r"@\w+|#\w+", "", text)

    return text


# Apply the clean_text function to the 'text' column and store the cleaned text in a new column 'cleaned_text'
DF_CLEAN['cleaned_text'] = DF_CLEAN['text'].apply(text_clean)

# Function to perform sentiment analysis using VADER


def get_polarity_vader(text):
    return SentimentIntensityAnalyzer().polarity_scores(text)['compound']

# Function to perform sentiment analysis using TextBlob


def get_polarity_textblob(text):
    return TextBlob(text).sentiment.polarity


# Apply sentiment analysis using VADER and store the polarity output in a new column 'VaderPolarity'
DF_CLEAN['VaderPolarity'] = DF_CLEAN['cleaned_text'].apply(get_polarity_vader)

# Apply sentiment analysis using TextBlob and store the polarity output in a new column 'TBPolarity'
DF_CLEAN['TBPolarity'] = DF_CLEAN['cleaned_text'].apply(get_polarity_textblob)

DF_CLEAN['VaderSentiment'] = ''
DF_CLEAN.loc[DF_CLEAN.VaderPolarity >= 0.05, 'VaderSentiment'] = 'POSITIVE'
DF_CLEAN.loc[DF_CLEAN.VaderPolarity.between(
    -0.05, 0.05, inclusive='left'), 'VaderSentiment'] = 'NEUTRAL'
DF_CLEAN.loc[DF_CLEAN.VaderPolarity < -0.05, 'VaderSentiment'] = 'NEGATIVE'

DF_CLEAN['TBSentiment'] = ''
DF_CLEAN.loc[DF_CLEAN.TBPolarity >= 0, 'TBSentiment'] = 'POSITIVE'
DF_CLEAN.loc[DF_CLEAN.TBPolarity == 0, 'TBSentiment'] = 'NEUTRAL'
DF_CLEAN.loc[DF_CLEAN.TBPolarity < 0, 'TBSentiment'] = 'NEGATIVE'
##########################################################################


####################### Calculation of Engagement Rate ####################
DF_ENG['total_tweets'] = DF_ENG['total_tweets'].str.replace(
    ',', '').astype(int)
DF_ENG['num_following'] = DF_ENG['num_following'].str.replace(
    ',', '').astype(int)
DF_ENG['num_followers'] = DF_ENG['num_followers'].str.replace(
    ',', '').astype(int)
DF_ENG['total_likes'] = DF_ENG['total_likes'].str.replace(',', '').astype(int)

DF_ENG['timestamp'] = pd.to_datetime(DF_ENG["timestamp"])
DF_ENG["Date"] = DF_ENG['timestamp'].dt.date
DF_SORTED = DF_ENG.sort_values(by='timestamp', ascending=False)
DAYS_10 = DF_SORTED['timestamp'].dt.date.unique()[:10]
DF_ENG_NEW = DF_SORTED[DF_SORTED['timestamp'].dt.date.isin(DAYS_10)].copy()

# Calculate engagement rate as a percentage
DF_ENG_NEW['engagement_rate'] = (DF_ENG_NEW['likes'] + 
    DF_ENG_NEW['retweets'] + DF_ENG_NEW['replies'] + DF_ENG_NEW['quotes']) / (len(DF_ENG))
# st.write(DF_ENG_NEW['engagement_rate'])

# Calculate the total engagement and total number of followers for each date
engagement_by_date = DF_ENG_NEW.groupby('Date').agg(
    {'engagement_rate': 'sum', 'num_followers': 'sum'}).reset_index()

# Calculate the average engagement rate per follower
engagement_by_date['avg_engagement_rate_per_follower'] = (engagement_by_date['engagement_rate'] / \
    513100) * 100

#########################################################################


################### Calculation of Impression Rate ######################
DF_HASH['timestamp'] = pd.to_datetime(DF_HASH["timestamp"])
DF_HASH["Date"] = DF_HASH['timestamp'].dt.date
DF_SORTED = DF_HASH.sort_values(by='timestamp', ascending=False)
DAYS_10 = DF_SORTED['timestamp'].dt.date.unique()[:10]
DF_HASH_NEW = DF_SORTED[DF_SORTED['timestamp'].dt.date.isin(DAYS_10)].copy()
DF_IMP = DF_HASH_NEW.copy()

# Calculate impressions by multiplying followers with the number of tweets
DF_IMP['Impression'] = DF_IMP.groupby(['username', 'Date'])['num_followers'].transform('sum')

# Sum the impressions for each date
impressions_by_date = DF_IMP.groupby('Date')['Impression'].sum()

# Print the impressions of the hashtag for each date
total_impact = pd.DataFrame(impressions_by_date).reset_index()

# total_impact = username_counts.groupby(
    # 'Date')['impact'].sum().reset_index(name='total_impact')
#########################################################################


########################## M&M Dashboard ############################
st.title("M&M Dashboard")
# st.markdown('<hr style="border: 1px solid black;">', unsafe_allow_html=True)
st.subheader("Average Response Time")
#### Response Time #####
col1_1, col1_2 = st.columns(2)

with col1_1:
    st.markdown("**Average reply:**")
    st.markdown(f"<ul><li style='color: red; font-size: 20px;'><b><i>{round(average_seconds, 2)}</i></b> seconds</li><li style='color: red; font-size: 20px;'><b><i>{round(average_days, 2)}</i></b> days</li></ul>", unsafe_allow_html=True)

with col1_2:  
        fig_time, ax = plt.subplots()
        ax.bar(df["Time Range"], df["Number of Responses"], color='blue')

        ax.set_xlabel('Response Time Range')
        ax.set_ylabel('Number of Responses')
        ax.set_title('M&M Response Time Distribution')
        ax.tick_params(axis='x', rotation=45)

        st.pyplot(fig_time)


##### Engagement & Sentiment Graph ######
container1 = st.container()
col1, col2 = st.columns(2)

with container1:
    with col1:
        st.subheader("Engagement Rate")
        fig, ax = plt.subplots()
        ax.plot(np.array(engagement_by_date['Date']), np.array(
            engagement_by_date['avg_engagement_rate_per_follower']))
        ax.set_xlabel('Date')
        ax.set_ylabel('Engagement')
        # ax.set_title('Total M&M Engagement over Time')
        ax.tick_params(axis='x', rotation=45)

        # Display the line chart using st.pyplot()
        st.pyplot(fig)

    with col2:
        st.subheader("Sentiment Analysis")
        X = ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
        X_axis = np.arange(len(X))

        fig, ax = plt.subplots()  # Create a figure and axis object
        ax.bar(X_axis - 0.2, DF_CLEAN['VaderSentiment'].value_counts(),
               width=0.4, label='VaderSentiment')
        ax.bar(
            X_axis + 0.2, DF_CLEAN['TBSentiment'].value_counts(), width=0.4, label='TBSentiment')

        ax.set_xticks(X_axis)
        ax.set_xticklabels(X)
        ax.set_ylabel("Counts")
        # ax.set_title("Customer Sentiment")
        ax.legend()

        # Display the figure using st.pyplot()
        st.pyplot(fig)


st.markdown('<hr style="border: 1px solid black;">', unsafe_allow_html=True)

###### Reach & Impression Graph #######
st.markdown('<h5 style="text-align:center; background-color:black; color:white;">Hashtag: <b>#MMSIceCreamSweepstakes</b></h5>', unsafe_allow_html=True)

container2 = st.container()
col3, col4 = st.columns(2)

with container2:  
    with col3:
        st.subheader("Reach Rate")
        # Create the Matplotlib figure
        fig, ax = plt.subplots()
        ax.plot(np.array(followers_by_date.index),
                np.array(followers_by_date.values))
        ax.set_xlabel('Date')
        ax.set_ylabel('Reach')
        # ax.set_title('Reach over Date')
        ax.tick_params(axis='x', rotation=45)

        # Display the figure using st.pyplot()
        st.pyplot(fig)

    with col4:
        st.subheader("Impression Rate")
        fig, ax = plt.subplots()
        ax.plot(np.array(total_impact['Date']),
                np.array(total_impact['Impression']))
        ax.set_xlabel('Date')
        ax.set_ylabel('Impression')
        # ax.set_title('Impression over Date')
        ax.tick_params(axis='x', rotation=45)

        # Display the line chart using st.pyplot()
        st.pyplot(fig)

###################################################################################################
