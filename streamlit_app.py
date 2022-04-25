from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# Set Widescreen format
st.set_page_config(layout="wide")

"""
# SPODIO RSS Feed Monitor
Tracks and aggregates Sports RSS Feeds.

TODO:
Add analytics to determine which posts are most relivent/timely


Note: Links in Summaries may not work (sometime RSS feedsmess them up)
    
# Most recent 100 RSS posts
"""
# In[]
Post_df = pd.read_excel('Post_History.xlsx')
Post_df = Post_df.sort_values('published',ascending=False)
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


# In[]
components.html(Table_Styler.to_html(),width=1900, height=1000, scrolling=True)

