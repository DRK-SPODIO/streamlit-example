#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:54:45 2022

@author: spodionew
"""


# imports
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser
from dateutil import tz
import feedparser



# =============================================================================
# Load RSS Links and gather new posts
# =============================================================================
# Load RSS Links and convert to list
links_RSS_df = pd.read_excel('RSS_Links.xlsx')
links_RSS = links_RSS_df['RSS_links'].tolist()

# Load RSS Posts from each link
print('Getting RSS Posts')
start = datetime.now()
print('Started: ', start)
feeds = [[datetime.now(), url, feedparser.parse(url)['entries']] for url in links_RSS]
end = datetime.now()
print('Completed: ', end)
print('Time to complete', end-start)


# Gather posts from each Feed into a single table.
Post_Counts = []
Feeds_DF = pd.DataFrame()
print('Processing feeds...')
for feed in feeds:
    Feed_DF = pd.DataFrame(feed[2])
    Post_Counts.append(len(feed[2]))
    Feeds_DF = Feeds_DF.append(Feed_DF)

# In[]
# =============================================================================
# Porcess posts and add to existing Post History
# =============================================================================
df = Feeds_DF[['published', 'title', 'link', 'summary', 'author']].copy() # pass data to init

# Fix dates and make Universal time
Dates = df.published.tolist()
Dates_Clean = [x.split(', ')[1] if ', ' in x else x for x in Dates]
Dates_Clean = [x if 'HH:' not in x else x.split(' HH:')[0] for x in Dates_Clean]
Dates_Clean = [np.nan if x == '' else x for x in Dates_Clean]
Dates_Clean = [dateutil.parser.parse(str(x)) if str(x) != 'nan' else x for x in Dates_Clean]
Dates_Clean = [x.astimezone(tz.UTC) if str(x) != 'nan' else x for x in Dates_Clean]
Dates_Clean = [x.replace(tzinfo=None) if str(x) != 'nan' else x for x in Dates_Clean]
df['published'] = Dates_Clean

# Select columns for Posts File
Post_df = df[['published', 'author', 'title', 'link', 'summary']].copy()
# Load older posts, update with new info, delete older records.
Old_Posts_df = pd.read_excel('Post_History.xlsx')

# Add Old Posts, clean up Post History
Post_df = Post_df.append(Old_Posts_df)
Post_df['author'] = Post_df['author'].fillna('No author')
Post_df['link'] = Post_df['link'].fillna('No Link')
Post_df['summary'] = Post_df['summary'].fillna('No Summary')

Post_df['summary'] = [x if len(x) > 4 else 'No Summary' for x in Post_df['summary'].tolist()]
Post_df['link'] = [x if len(x) > 1 else 'No Link' for x in Post_df['link'].tolist()]
Post_df['author'] = [x if len(x) > 1 else 'No Author' for x in Post_df['author'].tolist()]

# Remove Links from end of RSS feeds that used language from another link shorthand
# on a non-related post (caused mixed language to be sent to the LDA model and duplicate posts)
Post_df['summary'] = [x.split('<p>The post <a')[0] for x in Post_df['summary'].tolist()]
Post_df['link'] = Post_df['link'].fillna('No Link')
Post_df = Post_df[~Post_df['link'].str.contains('https://blog.beta.nostragamus.in')]

# Fix posts that include views creating duplicates
Post_df['summary'] = Post_df['summary'].fillna('No Summary')
Post_df['summary'] = ['<p>'+x.split('&#160;views today')[1] if '&#160;views today ' in x else x for x in Post_df.summary.tolist()]
Post_df['summary'] = [x.strip() for x in Post_df.summary.tolist()]

# Replace Missing Values Again
Post_df['summary'] = [x if len(x) > 4 else 'No Summary' for x in Post_df['summary'].tolist()]
Post_df['link'] = [x if len(x) > 1 else 'No Link' for x in Post_df['link'].tolist()]
Post_df['author'] = [x if len(x) > 1 else 'No Author' for x in Post_df['author'].tolist()]

Post_df['author'] = Post_df['author'].fillna('No author')
Post_df['link'] = Post_df['link'].fillna('No Link')
Post_df['summary'] = Post_df['summary'].fillna('No Summary')

# Remove NaT Values
Post_df = Post_df.dropna()
# Sort Posts by Publication Date
Post_df = Post_df.sort_values('published', ascending=False).reset_index(drop=True)
# Drop Duplicates
Post_df = Post_df.drop_duplicates(subset=['published','title', 'link'], keep='first').copy()

# Delete RSS.app notice that trial expired.
Post_df = Post_df[~Post_df['summary'].str.contains('Your trial has expired.')]

# Delete trailing text from Sport Insider RSS Posts
Search_String = 'Use of this feed is for personal non-commercial use only. '
Post_df['summary'] = [str(doc).split(Search_String)[0] for doc in Post_df['summary'].tolist()]

# Write to excel File
print('New Posts:', len(Post_df)-len(Old_Posts_df))
print('Saving to Excel file...')
Post_df.to_excel('Post_History.xlsx', index=False)
print('Data update Complete')

