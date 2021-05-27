# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:39:17 2021

@author: ludov
"""


import tweepy
import pandas as pd
import texthero as hero



'--------------------// Get tweets from twitter developer account //-------------------'

def get_all_tweets(screen_name
                   ,consumer_key = '0Jl2dfQ7J5OMQHSXddjYQXj97'
                   , consumer_secret= 'DQLR9NU4lIKUhOHsm4mlHGExRESUBZxjtru84F3yZgm47mCyC0'
                   , access_key= '1361377080043331593-PJbVm25azdPoczvM7JYdchpnMPLnXa'
                   , access_secret= 'br67qYffuhgNdMicXngkAaJzWoxY8mNxmyu3wOyXJ66HK'
                   ):
    #Twitter only allows access to a users most recent 3240 tweets with this method
    
    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    
    #initialize a list to hold all the tweepy Tweets
    alltweets = []  
    
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)



    #save most recent tweets
    alltweets.extend(new_tweets)
    
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        
        
        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
        
        #save most recent tweets
        alltweets.extend(new_tweets)
        
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
           
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in alltweets]
    tweets_df = pd.DataFrame(outtweets, columns = ['time', 'datetime', 'text'])

    return tweets_df
 


'----------------------// Tweets extraction according to selected sources //----------------------------'

def get_options_flow():


    ss = get_all_tweets(screen_name ="SwaggyStocks")
    uw = get_all_tweets(screen_name ="unusual_whales")
    
    ss['source'] = 'swaggyStocks'
    ss['text'] = hero.remove_urls(ss['text'])
    ss['text'] = [n.replace('$','') for n in ss['text']]
    
    
    uw['source'] = 'unusual_whales'
    uw['text'] = hero.remove_urls(uw['text'])
    uw['text'] = [n.replace('$','') for n in uw['text']]
    uw['text'] = [n.replace(':','') for n in uw['text']]
    uw['text'] = [n.replace('\n',' ') for n in uw['text']]
    uw['text'] = [n.replace('  ',' ') for n in uw['text']]
    
    
        
    tweets = pd.concat([ss, uw])


    return (tweets)




twitter_posts=get_options_flow()
