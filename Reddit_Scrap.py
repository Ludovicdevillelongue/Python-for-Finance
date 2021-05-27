# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:50:29 2021

@author: ludov
"""

import pandas as pd
import praw


'----------------------return a dataframe for the newest reddit posts-----------------------------'


def get_reddit(cid= '-b4qa9lww0zR6A', csec= 'pok8UWJjAyq9ER3_04Is3RPf8DsXVw', uag= 'Chemical_Tax_7919', subreddit='wallstreetbets'):
    reddit = praw.Reddit(client_id= cid, client_secret= csec, user_agent= uag)


#get posts from wallstreetbets
    posts = reddit.subreddit('wallstreetbets').hot(limit=1000)


    p = []
    for post in posts:
        p.append([post.title, post.score, post.selftext])
    posts_df = pd.DataFrame(p,columns=['title', 'score', 'post'])
    return posts_df

reddit_posts_wsb=get_reddit()

