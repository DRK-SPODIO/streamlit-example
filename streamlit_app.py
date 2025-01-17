from collections import namedtuple
import altair as alt
import math
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

st.set_page_config(layout="wide") # Set page to wide view

# In[]
# Post_df = pd.read_excel('Post_History.xlsx')
# Post_df = Post_df[~Post_df['summary'].isin(['No Summary'])]
# Post_df = Post_df.sort_values('published',ascending=False)



# In[]
# Set Widescreen format

def Comma_Format(number):
    return str(("{:,}".format(number)))

# Header & Description
"""
# SPODIO RSS Feed Monitor
Tracks and aggregates Sports RSS Feeds.
    
"""


# Display_df = Post_df.head(100).copy()

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
 

# Display_df['link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if not pd.isna(x) else 'No Link' for x in  Display_df['link'].tolist()]
# Display_df['summary'] = Display_df['summary'].fillna('No Summary')
# Display_df = Display_df.reset_index(drop=True)

# Clean up posts that include non-working links
# Display_df['summary'] = Display_df['summary'].str.replace('<p>The post <a href="https://www.sportsnet.ca/feed/" rel="nofollow">Why Stripling fits perfectly into Blue Jays&#8217; six-man pitching rotation</a> appeared first on <a href="https://www.sportsnet.ca" rel="nofollow">Sportsnet.ca</a>.</p>','',regex=False)

# Table_Styler = Display_df.style.set_table_styles(styles).hide_index()

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




st.cache(suppress_st_warning=True)
def Get_Data():
    return pd.read_excel('Doc2Vec Results.xlsx')

"""
# SPODIO RSS Feed Analytics: Most Relivent 25 RSS posts from Topic (Top2Vec Model V3.0)

Topics are sorted by most written about, so Topic Numbers can change over time.

"""

with st.spinner('Loading Data...'):
    DNN_Model_df_Data = Get_Data()
st.success('Data Loaded')

# Post Count Widget
Post_Count = Comma_Format(len(DNN_Model_df_Data))+' Posts'
Age = pd.DataFrame()
Age['Age'] = [datetime.utcnow() - x for x in DNN_Model_df_Data['published'].tolist()] # Calculate Post age
Age = Age['Age'] / np.timedelta64(1, 'h')  # Convert to hours
Age = Age.iloc[[x < 24 for x in Age]]
New_Posts = Comma_Format(len(Age))+' Posts in last 24 hours'
st.metric(label="Number of Posts", value=Post_Count, delta=New_Posts)

def Sort_df():
    DNN_Model_df = DNN_Model_df_Data[DNN_Model_df_Data.Topic == 'Topic '+str(Topic_Selector)]
    DNN_Model_df = DNN_Model_df.sort_values(by='Rel_Age_Score',axis=0, ascending=False).reset_index(drop=True)
    # Get top 25 Posts
    DNN_Model_df = DNN_Model_df.head(25)
    # Format Links
    DNN_Model_df['Link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if x != "No Link" and not pd.isna(x) else 'No Link' for x in DNN_Model_df['link'].tolist()]
    # Format Table
    DNN_Model_df = DNN_Model_df[['published', 'author', 'Link', 'title', 'summary']]
    Table_DNN_Styler = DNN_Model_df.style.set_table_styles(styles).hide_index()    
    # Display Table
    return Table_DNN_Styler


Topic_Selector = st.slider('Topic Selection', min_value=1, max_value=150, value=1, step=1, help='Select a Topic Group', on_change=Sort_df)

components.html(Sort_df().to_html(),width=1500, height=1000, scrolling=True)

# """
# Experimental Smart Search
# """
# from top2vec import Top2Vec
# model = Top2Vec.load('2205270814_Top2VecModel.mod')
# Search_Words = st.text_input('Search Terms','Max Verstappen')
# Search_Words = Search_Words.split()
# documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=Search_Words, num_docs=20)
# Search_Results = DNN_Model_df_Data[DNN_Model_df_Data.doc_id.isin(document_ids)].copy()

# # Format Links
# Search_Results['Link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if x != "No Link" and not pd.isna(x) else 'No Link' for x in Search_Results['link'].tolist()]
# # Format Table
# Sort_by = 'published'
# Search_Results = Search_Results.sort_values(by=Sort_by,axis=0, ascending=False).reset_index(drop=True)
# Search_Results = Search_Results[['published', 'author', 'Link', 'title', 'summary']]
# Table_Search_Results = Search_Results.style.set_table_styles(styles).hide_index().to_html()

# components.html(Table_Search_Results,width=1400, height=1000, scrolling=True)


# In[]
# Print Tops closest to Keyword Search;
# Keywords = ["NFL", "Draft", "QB"]
# documents, document_scores, document_ids = model.search_documents_by_keywords(keywords=Keywords, num_docs=10)
# for doc, score, doc_id in zip(documents, document_scores, document_ids):
#     print(f"Document: {doc_id}, Score: {score}")
#     print("-----------")
#     print(cleanhtml(papers['title'].tolist()[doc_id]))
#     print(cleanhtml(papers['summary'].tolist()[doc_id]))
#     print("-----------")
#     print()


