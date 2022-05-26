from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(layout="wide")

# In[]
Post_df = pd.read_excel('Post_History.xlsx')
Post_df = Post_df[~Post_df['summary'].isin(['No Summary'])]
Post_df = Post_df.sort_values('published',ascending=False)



# In[]
# Set Widescreen format

# Header & Description
"""
# SPODIO RSS Feed Monitor
Tracks and aggregates Sports RSS Feeds.

TODO:
Add analytics to determine which posts are most relivent/timely

Note: Links in Summaries may not work (sometime RSS feeds mess them up)
    
"""
# Post Count Widget
Post_Count = str(len(Post_df))+' Posts'
Age = pd.DataFrame()
Age['Age'] = [datetime.utcnow() - x for x in Post_df['published'].tolist()] # Calculate Post age
Age = Age['Age'] / np.timedelta64(1, 'h')  # Convert to hours
Age = Age.iloc[[x < 24 for x in Age]]
New_Posts = str(len(Age))+' Posts in last 24 hours'
st.metric(label="Number of Posts", value=Post_Count, delta=New_Posts)

Display_df = Post_df.head(100).copy()

styles = [
    dict(selector="tr:hover",
                props=[("background", "#f4f4f4")]),
    dict(selector="th", props=[("color", "#fff"),
                               ("border", "1px solid #eee"),
                               ("padding", "12px 35px"),
                               ("border-collapse", "collapse"),
                               ("background", "#1F2441"),
                               ("text-transform", "capitalize"),
                               ("font-size", "16px")
                               ]),
    dict(selector="td", props=[("color", "#605ESE"),
                               ("border", "1px solid #eee"),
                               ("padding", "12px 35px"),
                               ("border-collapse", "collapse"),
                               ("font-size", "15px")
                               ]),
    dict(selector="table", props=[
                                    ("font-family" , 'Arial'),
                                    ("margin" , "25px auto"),
                                    ("border-collapse" , "collapse"),
                                    ("border" , "1px solid #eee"),
                                    ("border-bottom" , "2px solid #00cccc"),                                    
                                      ]),
    dict(selector="caption", props=[("caption-side", "bottom")])
]

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
#st.set_page_config(layout='wide')  # Set page to wide view

Display_df['link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if not pd.isna(x) else 'No Link' for x in  Display_df['link'].tolist()]
Display_df['summary'] = Display_df['summary'].fillna('No Summary')
Display_df = Display_df.reset_index(drop=True)

# Clean up posts that include non-working links
Display_df['summary'] = Display_df['summary'].str.replace('<p>The post <a href="https://www.sportsnet.ca/feed/" rel="nofollow">Why Stripling fits perfectly into Blue Jays&#8217; six-man pitching rotation</a> appeared first on <a href="https://www.sportsnet.ca" rel="nofollow">Sportsnet.ca</a>.</p>','',regex=False)

Table_Styler = Display_df.style.set_table_styles(styles).hide_index()

# """# 100 Most recent RSS posts """
# components.html(Table_Styler.to_html(),width=2400, height=1000, scrolling=True)


# In[]

# Test to put Topic Model HTML into StreamLit App (not working)
#components.iframe("/Topic_Model.html", width=2400, height=800)

# """
# # SPODIO RSS Feed Analytics
    
# # Most Relivent 100 RSS posts (LDA Model V2.0)
# TODO: Add topic selection (currently most recent relivent posts to all topics)
# """

# Rel_df = pd.read_excel('Post_Analytics.xlsx')
# Rel_df = Rel_df[~Rel_df['Post Text'].isin(['No Summary'])]
# Rel_df = Rel_df[['Token_Score_Aged','Date','Site', 'Link', 'Title', 'Post Text']]
# Rel_df = Rel_df.sort_values(by='Token_Score_Aged', ascending=False).reset_index(drop=True)

# Display_Rel_df = Rel_df.head(100).copy()
# Display_Rel_df['Link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if not pd.isna(x) else 'No Link' for x in  Display_Rel_df['Link'].tolist()]
# Display_Rel_df = Display_Rel_df.reset_index(drop=True)

# Table_Rel_Styler = Display_Rel_df.style.set_table_styles(styles).hide_index()

# components.html(Table_Rel_Styler.to_html(),width=2400, height=1000, scrolling=True)

# In[]


"""
# SPODIO RSS Feed Analytics
    
# Most Relivent 25 RSS posts from Top Selector (Top2Vec Model V3.0)
TODO: Add topic selection (currently most recent relivent posts to all topics)
"""
Topic_Selector = st.slider('Topic Selection (lower Topic number indicates more posts in Topic)', min_value=1, max_value=150, value=1, step=1, help='Select a Topic Group', on_change=None)


DNN_Model_df = pd.read_excel('Doc2Vec Results.xlsx')
DNN_Model_df = DNN_Model_df[DNN_Model_df.Topic == 'Topic '+str(Topic_Selector)]
DNN_Model_df = DNN_Model_df.sort_values(by='Rel_Age_Score',axis=0, ascending=False).reset_index(drop=True)

# Get top 25 Posts
DNN_Model_df = DNN_Model_df.head(25)
# Format Links
DNN_Model_df['Link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if not pd.isna(x) else 'No Link' for x in  DNN_Model_df['link'].tolist()]

# Format Table
DNN_Model_df = DNN_Model_df[['published', 'author', 'Link', 'title', 'summary']]
Table_DNN_Styler = DNN_Model_df.style.set_table_styles(styles).hide_index()

# Display Table
components.html(Table_DNN_Styler.to_html(),width=2400, height=1000, scrolling=True)
