import json
import csv
import pandas as pd
import tweepy
import numpy as np
from tweepy import OAuthHandler

consumer_key = "3p55u8EwlzNobuWD7U5nXIeYl"
consumer_secret = "aJK4V435vpFEpc8QzSpZrC6eowgU8htL7ASA1lAymLuIoDGFlK"
access_token = "622342987-5HfvPt0HNKmGOraYVRiovrMxpQUpxiYCYhqu4pyN"
access_token_secret = "AkA4gmoPqXN6LFlkpr62NAkksCkNGXMz0DwdKIyNqhftu"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# try:
#     api.verify_credentials()
#     print("Authentication Successful")
# except:
#     print("Authentication Error")

# # TEST YOUR ACCESS
# api.home_timeline()


############################# Snickers ######################################
username_snickers = "@SNICKERS"

def getTweets(username, c, filename):
    userTweets = api.user_timeline(screen_name = username, count=c)
    output = [[twit.id, twit.text, twit.created_at] for twit in userTweets]

    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(output)
    pass

getTweets(username_snickers, 10000, "snickers_tweets.csv")
############################################################################


############################# Cadbury UK ###################################
words = '#DontSearchTwirlMint'
date_since = '2023-06-20'
numtweet = 1000

db = pd.DataFrame(columns=[ 'createdate',
                            'username',
                            'followers',
                            'text'])

# We are using .Cursor() to search through twitter for the required tweets.
# The number of tweets can be restricted using .items(number of tweets)
tweets = tweepy.Cursor(api.search_tweets,
                      words, lang="en",
                      since_id=date_since,
                      tweet_mode='extended').items(numtweet)

# .Cursor() returns an iterable object. Each item in the iterator has various attributes that you can access to
# get information about each tweet
list_tweets = [tweet for tweet in tweets]

# Counter to maintain Tweet Count
i = 1

# we will iterate over each tweet in the list for extracting information about each tweet
for tweet in list_tweets:
    createddate = tweet.created_at
    username = tweet.user.screen_name
    followers = tweet.user.followers_count
    following = tweet.user.friends
    try:
            text = tweet.retweeted_status.full_text
    except AttributeError:
            text = tweet.full_text

    # Here we are appending all the
    # extracted information in the DataFrame
    ith_tweet = [ createddate ,username,
                followers,text]
    db.loc[len(db)] = ith_tweet
    i = i+1

# reference : https://www.geeksforgeeks.org/extracting-tweets-containing-a-particular-hashtag-using-python/

filename = 'hashtag_tweets.csv'

# # we will save our database as a CSV file.
db.to_csv(filename)

############################################################################

getTweets("@KITKAT", 10000, "kidkat_tweets.csv")
getTweets("@Smarties", 10000, "SmartiesUKI_tweets.csv")
getTweets("@mmschocolate", 10000, "M&M_tweets.csv")
