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

import re


# Load RSS Links and convert to list
links_RSS_df = pd.read_excel('RSS_Links.xlsx')
links_RSS = links_RSS_df['RSS_links'].tolist()

# Load RSS Posts from each link
print('Getting RSS Posts')
start = datetime.now()
feeds = [[datetime.now(), url, feedparser.parse(url)['entries']] for url in links_RSS]
end = datetime.now()
print('Time to complete', end-start)

# Gather posts from each Feed into a single table.
Post_Counts = []
Feeds_DF = pd.DataFrame()
for feed in feeds:
    Feed_DF = pd.DataFrame(feed[2])
    Post_Counts.append(len(feed[2]))
    Feeds_DF = Feeds_DF.append(Feed_DF)

# In[]
df = Feeds_DF[['published', 'title', 'link', 'summary', 'author']].copy() # pass data to init

# Fix dates and make Universal time
import numpy as np
Dates = df.published.tolist()
Dates_Clean = [x.split(', ')[1] if ', ' in x else x for x in Dates]
Dates_Clean = [x if 'HH:' not in x else x.split(' HH:')[0] for x in Dates_Clean]
Dates_Clean = [np.nan if x == '' else x for x in Dates_Clean]
Dates_Clean = [dateutil.parser.parse(str(x)) if str(x) != 'nan' else x for x in Dates_Clean]
Dates_Clean = [x.astimezone(tz.UTC) if str(x) != 'nan' else x for x in Dates_Clean]
Dates_Clean = [x.replace(tzinfo=None) if str(x) != 'nan' else x for x in Dates_Clean]
df['published'] = Dates_Clean

# In[]

# Select columns for Posts File
Post_df = df[['published', 'author', 'title', 'link', 'summary']].copy()

# TODO: Load older posts, update with new info, delete older records.
Old_Posts_df = pd.read_excel('Post_History.xlsx')

Post_df.append(Old_Posts_df)
Post_df = Post_df.drop_duplicates(subset=['author', 'title', 'link', 'summary'], keep='first').copy()
# Fix missing authors
Post_df['author'] = Post_df['author'].fillna('No author')
# Fix dates for Excel
Post_df['published'] = Post_df.published.dt.date  # Depricated for posts that have no date
# Write to excel File
Post_df.to_excel('Post_History.xlsx', index=False)

# In[]



# In[]
# import sweetviz as sv
# my_report = sv.analyze(Post_df)
# my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"

