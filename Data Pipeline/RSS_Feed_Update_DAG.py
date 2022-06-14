import json
import os 
from datetime import datetime
import dateutil.parser
from dateutil import tz
import pandas as pd
import numpy as np
import feedparser
import urllib.parse

from airflow import DAG
#from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import PythonOperator

import snowflake.connector


 

#Create Default Airflow Arguments
default_args = {
    'owner': 'airflow',
    'retries': 0,
    'depends_on_past': False,    
    'start_date': datetime(2022, 6, 1),
    'email': ['Anwar.Hossain@Toptal.com'],
    'email_on_failure': True,
    'email_on_retry': False 
    
                                    
}

# Set Schedule: Run pipeline every 30 mins every day. 
schedule_interval = "*/30 * * * *"

# Define DAG: Set ID and assign default args and schedule interval
dag = DAG(
    dag_id='RSS_Posts_Extract_Load', 
    default_args=default_args, 
    schedule_interval=schedule_interval
    )


def Extract_Complete():
    print ("RSS extract, clean up and merging complete")
    

def RSS_Posts_Extract_Load():
    
    #Read Credential from Environment Variable
    env_var = json.loads(os.environ["Credential"])

    #Create Connection to snowflake Config Schema 
    snow_con_config = snowflake.connector.connect(
        user = env_var["user"],
        password = env_var["password"],
        account = env_var["account"],
        warehouse =  env_var["warehouse"],
        database = env_var["database"],
    schema = 'CONFIG' 
    )
    
    #Read links from RSS_Links table in snowflakes
    links_RSS = snow_con_config.cursor().execute("Select Link, Domain, Language, Country From RSS_FEED_MONITOR.CONFIG.RSS_LINKS").fetchall()
    print(f"# of Rows in Links Table: {len(links_RSS)}")
    #print(links_RSS)
      
    #########################################################################
    # Get RSS Posts by iterating over all links
    #########################################################################
    
    # Load RSS Posts from each link
    print('Getting RSS Posts')
    start = datetime.now()
    print('Started: ', start)
    feeds = [[datetime.now(), url[0],url[1],url[2], url[3], feedparser.parse(url[0])['entries']] for url in links_RSS]
    end = datetime.now()
    print('Completed: ', end)
    print('Time to complete', end-start)

        
    # Gather posts from each Feed into a single table.
    Post_Counts = []
    Feeds_DF = pd.DataFrame()
    print('Processing feeds...')
    for feed in feeds:
        Feed_DF = pd.DataFrame(feed[5])
        Feed_DF["Source"] = feed[1]
        Feed_DF["Domain"] = feed[2]
        Feed_DF["Language"] = feed[3]
        Feed_DF["Country"] = feed[4]
        Post_Counts.append(len(feed[5]))
        Feeds_DF = Feeds_DF.append(Feed_DF)
        
    df = Feeds_DF[['published', 'title', 'link', 'summary', 'author','Source','Domain','Language','Country']].copy() # pass data to init
    
    #########################################################################
    # Clean up data
    #########################################################################
    # Fix dates and make Universal time
    Dates = df.published.tolist()
    Dates_Clean = [x.split(', ')[1] if ', ' in x else x for x in Dates]
    Dates_Clean = [x if 'HH:' not in x else x.split(' HH:')[0] for x in Dates_Clean]
    Dates_Clean = [np.nan if x == '' else x for x in Dates_Clean]
 #   Dates_Clean = [x.replace('BST','+0100')  for x in Dates_Clean]
 #   Dates_Clean = [x.replace('PDT','-0700')  for x in Dates_Clean]
 #   Dates_Clean = [x.replace('EDT','-0400')  for x in Dates_Clean]
    
    Dates_Clean = [dateutil.parser.parse(str(x)) if str(x) != 'nan' else x for x in Dates_Clean]
    Dates_Clean = [x.astimezone(tz.UTC) if str(x) != 'nan' else x for x in Dates_Clean]
    Dates_Clean = [x.replace(tzinfo=None) if str(x) != 'nan' else x for x in Dates_Clean]
    
    df['published'] = Dates_Clean
    
    
    
    
    
    # Select columns for Posts File
    Post_df = df[['published', 'author', 'title', 'link', 'summary','Source','Domain','Language','Country']] 
    
    
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

    # Update Source to use specific link if available, otherwise keep the source RSS feed link
    Post_df.loc[Post_df["link"] != "No Link","Source"]  = Post_df.loc[Post_df["link"] != "No Link","link"]
    # Extract Domain information
    Post_df["Domain"]  = [urllib.parse.urlparse(url).hostname   for url in Post_df['Source'].tolist()]

    
    # Remove NaT Values
    Post_df = Post_df.dropna()
    # Sort Posts by Publication Date
    Post_df = Post_df.sort_values('published', ascending=False).reset_index(drop=True)
    # Drop Duplicates
    Post_df = Post_df.drop_duplicates(subset=['published','title', 'Source', 'summary'], keep='first').copy()
    
    # Delete RSS.app notice that trial expired.
    Post_df = Post_df[~Post_df['summary'].str.contains('Your trial has expired.')]
    
    # Delete trailing text from Sport Insider RSS Posts
    Search_String = 'Use of this feed is for personal non-commercial use only. '
    Post_df['summary'] = [str(doc).split(Search_String)[0] for doc in Post_df['summary'].tolist()]
   

        
    final_df = Post_df[['published', 'author', 'title', 'link', 'summary','Source','Domain','Language','Country']]
    
    
    #########################################################################
    # Merge new data to database
    #########################################################################
    table = "RSS_POST_History"
    
    insert_columns = ['published', 'author', 'title', 'link', 'summary','Source','Domain','Language','Country']
    update_columns = ['published', 'author', 'title', 'summary','Source','Domain','Language','Country']
    id_columns = ['Source','title', 'summary']
    
    if final_df.empty: 
        print(f'No rows to bulk upsert to {table}. Aborting.')
        return

    with snowflake.connector.connect(
            user = env_var["user"],
            password = env_var["password"],
            account = env_var["account"],
            warehouse =  env_var["warehouse"],
            database = env_var["database"],
            schema = 'PUBLIC'
	) as con:
        cur = con.cursor()
        print(f"BULK UPSERTING {final_df.shape[0]} {table.upper()} TO SNOWFLAKE")

		# convert to json
        stage = "RSS_Post_Stage"
        filename = f"{table}.json"
        final_df.to_json(filename,orient='records',lines=True,date_unit='s')
        filepath = os.path.abspath(filename)
        
        #Set timezone to UTC
        cur.execute("alter session set timezone='UTC';")
        #Saving the new data json file to Snowflake Staging area
        cur.execute(f"put file://{filepath} @{stage} overwrite=true;")
        #Merge new data to existing Snowflake history table
        cur.execute(f"""merge into {table}
						using (select {','.join([f'$1:{col} as {col}' for col in insert_columns])}
							from @{stage}/{filename}) t
						on ({' and '.join([f't.{col} = {table}.{col}' for col in id_columns])})
						when matched then
							update set {','.join([f'{col}=t.{col}' for col in update_columns])}
						when not matched then insert ({','.join(insert_columns)})
						values ({','.join([f't.{col}' for col in insert_columns])});""")
		# delete json file from the table stage
        cur.execute(f"remove @{stage}/{filename};")
        # delete the json file created
        os.remove(filename)
        print('\t...Merging and staging file clean up complete')
        
        cur.close()
        
     
    
t1 = PythonOperator(
    task_id='Read_RSS',
    provide_context=True,
    python_callable=RSS_Posts_Extract_Load,
    dag=dag)


 
t2 = PythonOperator(
    task_id='Complete',
    provide_context=True,
    python_callable=Extract_Complete,
    dag=dag)  

 

# Setting up Dependencies
t1 >> t2
