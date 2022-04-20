from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

"""
# SPODIO RSS Feed Monitor
Tracks and aggregates Sports RSS Feeds.

TODO:
Add analytics to determine which posts are most relivent/timely


Note: Links in Summaries may not work (sometime RSS feedsmess them up)
    
# Most recent 50 RSS posts
"""
# In[]
Post_df = pd.read_excel('Post_History.xlsx')
Post_df = Post_df.sort_values('published',ascending=False)
Display_df = Post_df.head(50).copy()

styles = [
    dict(selector="tr:hover",
                props=[("background", "#f4f4f4")]),
    dict(selector="th", props=[("color", "#fff"),
                               ("border", "1px solid #eee"),
                               ("padding", "12px 35px"),
                               ("border-collapse", "collapse"),
                               ("background", "#1F2441"),
                               ("text-transform", "uppercase"),
                               ("font-size", "18px")
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

Display_df['link'] = ['<a href="'+ str(x) +'" target = "_blank">Link to Post</a>' if not pd.isna(x) else 'No Link' for x in  Display_df['link'].tolist()]
Display_df['summary'] = Display_df['summary'].fillna('No Summary')
Display_df = Display_df.reset_index()
Disp = Display_df.style.set_table_styles(styles)


# In[]
components.html(Disp.to_html(),width=1200, height=800, scrolling=True)

