from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

"""
# SPODIO RSS Feed Monitor

Tracks 72 RSS Feeds and displays details here

Future
Add analytics to determine what posts are most relivent
    
"""



# In[]
Post_df = pd.read_excel('Post_History.xlsx')
Post_df = Post_df.sort_values('published',ascending=False)
Display_df = Post_df.head(10).copy()
Display_df['link'] = ['<a href="'+ str(x) +'">Link</a>' for x in Display_df['link'].tolist()]

# In[]

components.html(Display_df.style.to_html(),width=1200, height=800, scrolling=True)