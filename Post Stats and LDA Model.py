#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:13:04 2022

@author: DRK
"""


# imports
import pandas as pd
import numpy as np
import scipy.stats as st
from datetime import datetime
import re
from lxml import etree


# In[]

# Load Post History
Post_df = pd.read_excel('Post_History.xlsx')




print('Processing Post data...')


# Stopwords are words with little to no meaning value
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2",
                  "a3", "a4", "ab", "able", "about", "above", "abst", "ac",
                  "accordance", "according", "accordingly", "across", "act",
                  "actually", "ad", "added", "adj", "ae", "af", "affected",
                  "affecting", "affects", "after", "afterwards", "ag", "again",
                  "against", "ah", "ain", "ain't", "aj", "al", "all", "allow",
                  "allows", "almost", "alone", "along", "already", "also",
                  "although", "always", "am", "among", "amongst", "amoungst",
                  "amount", "an", "and", "announce", "another", "any",
                  "anybody", "anyhow", "anymore", "anyone", "anything",
                  "anyway", "anyways", "anywhere", "ao", "ap", "apart",
                  "apparently", "appear", "appreciate", "appropriate",
                  "approximately", "ar", "are", "aren", "arent", "aren't",
                  "arise", "around", "as", "a's", "aside", "ask", "asking",
                  "associated", "at", "au", "auth", "av", "available", "aw",
                  "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3",
                  "ba", "back", "bc", "bd", "be", "became", "because",
                  "become", "becomes", "becoming", "been", "before",
                  "beforehand", "begin", "beginning", "beginnings",
                  "begins", "behind", "being", "believe", "below", "beside",
                  "besides", "best", "better", "between", "beyond", "bi",
                  "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom",
                  "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but",
                  "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came",
                  "can", "cannot", "cant", "can't", "cause", "causes", "cc",
                  "cd", "ce", "certain", "certainly", "cf", "cg", "ch",
                  "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon",
                  "cn", "co", "com", "come", "comes", "con", "concerning",
                  "consequently", "consider", "considering", "contain",
                  "containing", "contains", "corresponding", "could",
                  "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr",
                  "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx",
                  "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de",
                  "definitely", "describe", "described", "despite", "detail",
                  "df", "di", "did", "didn", "didn't", "different", "dj", "dk", 
                  "dl", "do", "does", "doesn", "doesn't", "doing", "don", 
                  "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", 
                  "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", 
                  "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", 
                  "eight", "eighty", "either", "ej", "el", "eleven", "else", 
                  "elsewhere", "em", "empty", "en", "end", "ending", "enough", 
                  "entirely", "eo", "ep", "eq", "er", "es", "especially", 
                  "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", 
                  "every", "everybody", "everyone", "everything", "everywhere", 
                  "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", 
                  "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", 
                  "fill", "find", "fire", "first", "five", "fix", "fj", "fl", 
                  "fn", "fo", "followed", "following", "follows", "for", 
                  "former", "formerly", "forth", "forty", "found", "four", 
                  "fr", "from", "front", "fs", "ft", "fu", "full", "further", 
                  "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", 
                  "getting", "gi", "give", "given", "gives", "giving", "gj", 
                  "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", 
                  "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", 
                  "hadn't", "happens", "hardly", "has", "hasn", "hasnt", 
                  "hasn't", "have", "haven", "haven't", "having", "he", "hed", 
                  "he'd", "he'll", "hello", "help", "hence", "her", "here", 
                  "hereafter", "hereby", "herein", "heres", "here's", 
                  "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", 
                  "hid", "him", "himself", "his", "hither", "hj", "ho", "home", 
                  "hopefully", "how", "howbeit", "however", "how's", "hr", "hs",
                  "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", 
                  "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", 
                  "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", 
                  "i'm", "immediate", "immediately", "importance", "important", 
                  "in", "inasmuch", "inc", "indeed", "index", "indicate", 
                  "indicated", "indicates", "information", "inner", "insofar", 
                  "instead", "interest", "into", "invention", "inward", "io", 
                  "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd",
                  "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", 
                  "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", 
                  "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", 
                  "knows", "ko", "l", "l2", "la", "largely", "last", "lately", 
                  "later", "latter", "latterly", "lb", "lc", "le", "least", 
                  "les", "less", "lest", "let", "lets", "let's", "lf", "like", 
                  "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", 
                  "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", 
                  "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", 
                  "many", "may", "maybe", "me", "mean", "means", "meantime", 
                  "meanwhile", "merely", "mg", "might", "mightn", "mightn't", 
                  "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", 
                  "moreover", "most", "mostly", "move", "mr", "mrs", "ms", 
                  "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", 
                  "myself", "n", "n2", "na", "name", "namely", "nay", "nc", 
                  "nd", "ne", "near", "nearly", "necessarily", "necessary", 
                  "need", "needn", "needn't", "needs", "neither", "never", 
                  "nevertheless", "next", "ng", "ni", "nine", "ninety", "nl", 
                  "nn", "no", "nobody", "non", "none", "nonetheless", "noone", 
                  "nor", "normally", "nos", "not", "noted", "nothing", "novel", 
                  "now", "nowhere", "nr", "ns", "nt", "o", "oa", "ob", 
                  "obtain", "obtained", "obviously", "oc", "od", "of", "off", 
                  "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", 
                  "om", "omitted", "on", "once", "one", "ones", "only", "onto", 
                  "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", 
                  "otherwise", "ou", "ought", "our", "ours", "ourselves", 
                  "out", "outside", "over", "overall", "ow", "owing", "own", 
                  "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", 
                  "pages", "par", "part", "particular", "particularly", "pas", 
                  "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", 
                  "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", 
                  "po", "poorly", "possible", "possibly", "potentially", "pp", 
                  "pq", "pr", "predominantly", "present", "presumably", 
                  "previously", "primarily", "probably", "promptly", "proud", 
                  "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", 
                  "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", 
                  "rather", "rc", "rd", "re", "readily", "really", 
                  "reasonably", "recent", "recently", 
                  #"ref", "refs", 
                  "regarding", "regardless", "regards", "related", 
                  "relatively", "research", "research-articl", "respectively", 
                  "resulted", "resulting", "results", "rf", "rh", "ri", 
                  "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", 
                  "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", 
                  "same", "saw", "say", "saying", "says", "sc", "sd", "se", 
                  "sec", "second", "secondly", "section", "see", "seeing", 
                  "seem", "seemed", "seeming", "seems", "seen", "self", 
                  "selves", "sensible", "sent", "serious", "seriously", 
                  "seven", "several", "sf", "shall", "shan", "shan't", "she", 
                  "shed", "she'd", "she'll", "shes", "she's", "should", 
                  "shouldn", "shouldn't", "should've", "show", "showed", 
                  "shown", "showns", "shows", "si", "side", "significant", 
                  "significantly", "similar", "similarly", "since", "sincere", 
                  "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", 
                  "some", "somebody", "somehow", "someone", "somethan", 
                  "something", "sometime", "sometimes", "somewhat", 
                  "somewhere", "soon", "sorry", "sp", "specifically", 
                  "specified", "specify", "specifying", "sq", "sr", "ss", "st", 
                  "still", "stop", "strongly", "sub", "substantially", 
                  "successfully", "such", "sufficiently", "suggest", "sup", 
                  "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", 
                  "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", 
                  "tends", "tf", "th", "than", "thank", "thanks", "thanx", 
                  "that", "that'll", "thats", "that's", "that've", "the", 
                  "their", "theirs", "them", "themselves", "then", "thence", 
                  "there", "thereafter", "thereby", "thered", "therefore", 
                  "therein", "there'll", "thereof", "therere", "theres", 
                  "there's", "thereto", "thereupon", "there've", "these", 
                  "they", "theyd", "they'd", "they'll", "theyre", "they're", 
                  "they've", "thickv", "thin", "think", "third", "this", 
                  "thorough", "thoroughly", "those", "thou", "though", 
                  "thoughh", "thousand", "three", "throug", "through", 
                  "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", 
                  "tm", "tn", "to", "together", "too", "took", "top", "toward", 
                  "towards", "tp", "tq", "tr", "tried", "tries", "truly", 
                  "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", 
                  "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", 
                  "um", "un", "under", "unfortunately", "unless", "unlike", 
                  "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", 
                  "us", "use", "used", "useful", "usefully", "usefulness", 
                  "uses", "using", "usually", "ut", "v", "va", "value", 
                  "various", "vd", "ve", "ve", "very", "via", "viz", "vj", 
                  "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", 
                  "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", 
                  "way", "we", "wed", "we'd", "welcome", "well", "we'll", 
                  "well-b", "went", "were", "we're", "weren", "werent", 
                  "weren't", "we've", "what", "whatever", "what'll", "whats", 
                  "what's", "when", "whence", "whenever", "when's", "where", 
                  "whereafter", "whereas", "whereby", "wherein", "wheres", 
                  "where's", "whereupon", "wherever", "whether", "which", 
                  "while", "whim", "whither", "who", "whod", "whoever", 
                  "whole", "who'll", "whom", "whomever", "whos", "who's", 
                  "whose", "why", "why's", "wi", "widely", "will", "willing", 
                  "wish", "with", "within", "without", "wo", "won", "wonder", 
                  "wont", "won't", "words", "world", "would", "wouldn", 
                  "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", 
                  "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", 
                  "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", 
                  "you'll", "your", "youre", "you're", "yours", "yourself", 
                  "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", 
                  "zz"])


STOPWORDS.extend(['nan', 'sports', 'post', 'continue', 'reading', 'appeared', 
                  'saturday', 'sunday', 'monday', 'tuesday', 'wednesday', 
                  'thursday', 'friday', 'appeared', 'sportsnetca', 'read', 
                  'sportsnet', '2022', '22bet', 'summary', 'https', 'http', 
                  'outlook', 'twitter', 'facebook', 'from', 'subject', 're', 
                  'edu', 'use', 'summary', 'http', 'https', 'update', 'year', 
                  'view', 'views', 'site', 'lifestyle', 'technology', 'news', 
                  'ly', 'purpleptsd', 'live', 'stream', 'views', 'today',
                  'subscribe', 'bit', 'espn', 'espns', 'espnplus', 'youtubetv',  
                  'publish', 'click', 'from', 'subject', 're', 'edu', 'use', 
                  'summary', 'http',  'https', 'update', 'year', 'view', 
                  'views', 'site', 'lifestyle', 'technology', 'news', 'ly', 
                  'purpleptsd', 'live', 'stream', 'twitter', 'facebook', 
                  'youtube', 'app', 'prospectinsider',  'apps', 'fansided', 
                  'fanside', 'fansided', 'podcast',  'entertainment', 'watch', 
                  'update', 'updates', 'link', 'links', 'streamed', 'updated', 
                  'sky', 'rte', 'happen', 'april'])

STOPWORDS.extend(['league', 'game', 'team', 'player', 'season', 'time',
                 'play', 'playing', 'week', 'teams', 'games', 'players', 
                 'fan', 'fans'])

STOPWORDS = list(set(STOPWORDS))

# Depricated in favor or lxml approach
# Regular Expression to remove non-printable type
# CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

# def cleanhtml(raw_html):
#   cleantext = re.sub(CLEANR, ' ', raw_html)
#   cleantext = re.sub(' +', ' ', cleantext)
#   return cleantext




def cleanhtml(raw_html):
    response = etree.HTML(text=raw_html)
    return response.xpath('string(.)')


# Clean up Summaries, prep for NLP workflows.
CleanDisc = [cleanhtml(str(x)) for x in Post_df['summary'].tolist()]  # Remove HTML Tags
CleanDisc = [re.sub(r'[^\w\s]', '', x) for x in CleanDisc]  # Remmove punctuation
CleanDisc = [x.strip() for x in CleanDisc]  # cleanup leading/trailing white spaces
CleanDisc = [" ".join(y.lower() for y in x.split()) for x in CleanDisc]  # cleanup internal white spaces

# Clean up Titles, prep for NLP workflows, add to Summary
CleanTitle = [cleanhtml(str(x)) for x in Post_df['title'].tolist()]  # Remove HTML Tags
CleanTitle = [re.sub(r'[^\w\s]', '', x) for x in CleanTitle]  # Remmove punctuation
CleanTitle = [x.strip() for x in CleanTitle]  # cleanup leading/trailing white spaces
CleanTitle = [" ".join(y.lower() for y in x.split()) for x in CleanTitle]  # cleanup internal white spaces

Combined = []
for (Summary, Title) in zip(CleanDisc, CleanTitle):
    Combined.append(Title + ' ' + Summary)

CleanDisc = Combined
#  Collect Summaries by Date
CleanDisc_df = pd.DataFrame(CleanDisc,columns=['Summary'])
CleanDisc_df['Post Text'] = Post_df['summary'].tolist().copy()
CleanDisc_df['Title'] = Post_df['title'].tolist().copy()
CleanDisc_df['Author'] = Post_df['author'].tolist().copy()
CleanDisc_df['Link'] = Post_df['link'].tolist().copy()
CleanDisc_df['Datetime'] = Post_df['published'].tolist().copy()
# Remove Time from Datetime
CleanDisc_df['Date'] = CleanDisc_df['Datetime'].copy()
CleanDisc_df['Date'] = CleanDisc_df["Date"].dt.date

# Calculate Age of Posts (in hours)
CleanDisc_df['Age'] = [datetime.utcnow() - x for x in CleanDisc_df['Datetime'].tolist()] # Calculate Post age
CleanDisc_df['Age'] = CleanDisc_df['Age'] / np.timedelta64(1, 'h')  # Convert to hours
# Normalize Age
CleanDisc_df['Age'] =(CleanDisc_df['Age'] - CleanDisc_df['Age'].mean()) / CleanDisc_df['Age'].std()
# Take the inverse probability of Age
CleanDisc_df['Age'] = [1 - st.norm.cdf(x) for x in CleanDisc_df['Age'].tolist()]
# Get Top Level Domain (TLD) from RSS Link
CleanDisc_df['Site'] = [re.findall(r'://([\w\-\.]+)',x)[0] if len(x) > 7 and x != 'No Link' else x for x in Post_df['link'].tolist()]


""" Keywords that are driving the news may be of interest.  In the below experiment,
we split aggregate summary & title text into tokens, exclude english stopwords, then
find the count of each token for the day.  We then map these counts to tokens for each post.
The sum of token counts (Ex stopwords) may be a proxy for the posts that are driving Keyword
counts."""

# L
# # Download Wordnet through NLTK in python console:
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

def Wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

# Calculate Daily Token Counts (Ex Stopwords)
CleanDisc_Daily = CleanDisc_df.groupby(['Date'], as_index=False).agg({'Summary': ' '.join})
CleanDisc_Daily['Tokens'] = [x.split() for x in CleanDisc_Daily['Summary'].tolist()]
CleanDisc_Daily['Token_Values'] = [list(set(x)) for x in CleanDisc_Daily['Tokens'].tolist()]
CleanDisc_Daily['Token_Values'] = [[x for x in y if x not in STOPWORDS] for y in CleanDisc_Daily['Token_Values'].tolist()]

CleanDisc_Daily['Token_Values'] = [[lemmatizer.lemmatize(token, Wordnet_pos(token)) for token in post_text] for post_text in CleanDisc_Daily['Token_Values'].tolist()]


CleanDisc_Daily['Token_Counts'] = [[Tokens.count(x) if Tokens.count(x) > 1 else 0 for x in Token_list] for (Token_list, Tokens) in zip(CleanDisc_Daily['Token_Values'].tolist(), CleanDisc_Daily['Tokens'].tolist())]

# Normalize raw counts
Tokens_Counts_Normed = []
for Token_Set in CleanDisc_Daily['Token_Counts']:
    if len(Token_Set) > 1:
        mean = sum(Token_Set)/len(Token_Set)
        Stdev = np.sqrt(sum([((x - mean) ** 2) for x in Token_Set]) / len(Token_Set))
        if Stdev == 0 or not Stdev == Stdev:
            Stdev = 1
        Tokens_Counts_Normed.append([st.norm.cdf((x - mean) / Stdev) for x in Token_Set])
        
    else:
        Tokens_Counts_Normed.append([])

CleanDisc_Daily['Token_Counts'] = Tokens_Counts_Normed


# Find tokens per post
CleanDisc_df['Tokens'] = [x.split() for x in CleanDisc_df['Summary'].tolist()]
CleanDisc_df['Tokens'] = [[x for x in y if x not in STOPWORDS] for y in CleanDisc_df['Tokens'].tolist()]
CleanDisc_df = CleanDisc_df.merge(CleanDisc_Daily[['Token_Counts','Token_Values','Date']], how='left', on=['Date'])

# Remove Posts with no date
CleanDisc_df = CleanDisc_df[CleanDisc_df['Date'].notna()]

# Ensure a list is available for Toke Mapping (Replace NAN Token lists with an empty list)
CleanDisc_df['Token_Counts'].fillna("").apply(list)
CleanDisc_df['Token_Values'].fillna("").apply(list)


Post_Tokens = CleanDisc_df['Tokens'].tolist()[0]
Token_Counts = CleanDisc_df['Token_Counts'].tolist()[0]
Tokens = CleanDisc_df['Token_Values'].tolist()[0]
Post_Token_Map = [Token_Counts[Tokens.index(x)] if len(x) > 0 and x in Tokens else 0 for x in Post_Tokens]
Post_Token_Map = []
for index, row in CleanDisc_df.iterrows():
    Post_Tokens = row['Tokens']
    Token_Counts = row['Token_Counts']
    Token_Values = row['Token_Values']
    Post_Token_Map.append([row['Token_Counts'][row['Token_Values'].index(x)] if (x is not None) and (x in row['Token_Values']) else 0 for x in row['Tokens']])

# Map Daily Token Counts to Tokens extracted from each post
CleanDisc_df['Token_Map'] = Post_Token_Map
#Calculate Post Score as the sum of Token Counts present in each post
CleanDisc_df['Token_Score'] = [sum(x) for x in Post_Token_Map]
CleanDisc_df['Post_Token_Count'] = [len(x) for x in CleanDisc_df['Tokens'].tolist()]

CleanDisc_df['Token_Score'] = CleanDisc_df['Token_Score'] / CleanDisc_df['Post_Token_Count']

# DEBUG Plot Age KDE
#CleanDisc_df['Age'].plot.kde()

CleanDisc_df['Token_Score_Aged'] = CleanDisc_df['Token_Score'] * CleanDisc_df['Age']
# Remove Posts with Aged Token Score
CleanDisc_df = CleanDisc_df[CleanDisc_df['Token_Score_Aged'].notna()]
CleanDisc_df = CleanDisc_df.reset_index(drop=True)

# Gather final table and sort by Date and NLP Score
CleanDisc_Final = CleanDisc_df[['Token_Score_Aged','Age', 'Date', 'Site', 'Author', 'Title', 'Post Text', 'Link', 'Tokens', 'Token_Map']]
CleanDisc_Final = CleanDisc_Final.sort_values(by = ['Date', 'Token_Score_Aged'], ascending = [False, False], na_position = 'last')
CleanDisc_Final = CleanDisc_Final.reset_index(drop=True)


""" In the below experiment, we impliment a LDA model to find clusters of
words that relate to a topic in the RSS feed"""

import gensim
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import nltk

from nltk.corpus import stopwords
from pprint import pprint

# spacy for lemmatization
import spacy


stop_words = STOPWORDS
stop_words = list(set(stop_words))

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

print('Preprocessing Posts for LDA analysis...')
papers = pd.DataFrame()
papers['title'] = CleanDisc_Final['Title']
papers['paper_text'] = CleanDisc_Final['Title'] + ".  " + CleanDisc_Final['Post Text']

# Remove punctuation

papers['paper_text_processed'] =[cleanhtml(str(post)) for post in papers['paper_text']]

# Replace non-alpha numeric characters with spaces
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub('[^A-Za-z0-9]+', ' ', x))
# Remove excess whitespace
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: re.sub(' +', ' ', x))
papers['paper_text_processed'] = [doc.strip() for doc in papers['paper_text_processed'].tolist()]

# Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

data = papers.paper_text_processed.values.tolist()

data_words = list(sent_to_words(data))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])  # Don't exclude via POS tagging
        #texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# remove stop words
data_words = remove_stopwords(data_words)
# Remove empty lists
data_words = [x for x in data_words if len(x) > 0]

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)
data_words_trigrams = make_trigrams(data_words)



# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
data_words = data_lemmatized

papers['Tokens'] = data_words

# Create Dictionary
id2word = corpora.Dictionary(data_words)

# Create Corpus
texts = data_words

# Create a dictionary representation of the documents.
dictionary = Dictionary(texts)

# Filter out words that occur less than 20 documents or more than 50% of documents.
dictionary.filter_extremes(no_below=5, no_above=0.5)

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in texts]

# In[]
papers['corpus'] = corpus
# In[]
# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
print('Preprocessing for LDA model Complete')


# In[]
print('LDA Model is learning word clusters...')

""" In the following experiment, we impliment a Latient Dirichlet Allocation
model to estimate the distribution of contributions of words withiin an RSS
Post to a set number of topics.  In the way, words that are not strictly
applicable to specific topics """
# Train LDA model.
# Set training parameters.
num_topics = 30
chunksize = 11000
passes = 64
iterations = 500
eval_every = None  # Don't evaluate model perplexity, takes too much time.
#eval_every = 5

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

# In[]
""" In the following experiment, we evaluate the efficacy of a hierachical
Dirichlet Process to both characterize the number of topics contained in the
RSS post feed and nest topics within each other.  For example, posts on the 
NBA draft, would, ideally be clustered within all topics related to the NBA
and be seperated from topics related to the NFL or other leagues/sports."""

#TODO: Code fails due to underflow error; P(w|t) < machine precision

# print('Loading HDP Model...')
# from gensim.models import HdpModel


# model2 = HdpModel(corpus=corpus, 
#                   id2word=id2word, 
#                   max_chunks=None, 
#                   max_time=60 * 5, chunksize=chunksize, 
#                   var_converge=0.0001, outputdir=None, random_state=None)

# #                  kappa=1.0, tau=64.0, K=15, T=150, 
# #                  alpha=1, gamma=1, eta=0.01, scale=1.0, 

# top_topics2 = model2.get_topics()
# top_topics2_df = pd.DataFrame(top_topics2, columns=id2word.values())

import pyLDAvis
import pyLDAvis.gensim_models # don't skip this
import matplotlib.pyplot as plt

# vis2 = pyLDAvis.gensim_models.prepare(model2, corpus, dictionary)
# pyLDAvis.save_html(vis2, 'HDP_Topic_Model.html')
# Model_Runs = []
# for x in range(0,4):
#     model = LdaModel(
#         corpus=corpus,
#         id2word=id2word,
#         chunksize=chunksize,
#         alpha='auto',
#         eta='auto',
#         iterations=iterations,
#         num_topics=num_topics+x*5,
#         passes=passes,
#         eval_every=eval_every
#     )
#     top_topics = model.top_topics(corpus)
#     avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
#     print('Topic Count, Coherance',[num_topics + x*5, avg_topic_coherence])
#     Model_Runs.append([num_topics + x*5, avg_topic_coherence, model])

print('Learning Complete')
# In[]

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)
#pprint(top_topics)

Top_topics_df = pd.DataFrame(top_topics,columns=['Token_Map', 'Coherence'])
Top_topics_df = Top_topics_df[['Coherence', 'Token_Map']]

Topic_Word_Map = pd.DataFrame()
for index, row in Top_topics_df.iterrows():
    temp_df = pd.DataFrame(row.Token_Map,columns=['Weight', 'Word'])
    temp_df['Weight'] = temp_df['Weight'].astype(float)
    temp_df['Coherence'] = [row.Coherence for x in range(len(temp_df))]
    temp_df['Topic_Number'] = [index for x in range(len(temp_df))]
    Topic_Word_Map = Topic_Word_Map.append(temp_df)

Topic_Word_Map.reset_index(drop=True)
Topic_Matrix = model.get_topics()
Topic_Matrix = np.transpose(Topic_Matrix)


# Create Visual Report
vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary, R=50)
pyLDAvis.save_html(vis, 'Topic_Model.html')

# In[]
# Post Frequency by day and site
SiteCount_df = CleanDisc_df.groupby(['Site', 'Date']).size().reset_index(name="Posts")



Token_Dates = pd.DataFrame()

for Index, Row in CleanDisc_Final.iterrows():
    
    Tokens_Local = Row['Tokens']
    Dates = [Row['Date'] for x in range(len(Tokens_Local))]
    Token_Values_Local = Row['Token_Map']
    Local_df = {'Date': Dates,
                'Word': Tokens_Local,
                'Day_Value': Token_Values_Local}
    Local_df = pd.DataFrame(Local_df)
    Token_Dates = Token_Dates.append(Local_df)

Token_Dates.to_csv('News_Tokens.csv')
print("Post data processed.")

# In[]

train_vecs = []
train_ids = []
for i in range(len(corpus)):
    topics = (
        model.get_document_topics(corpus[i],
                                      minimum_probability=0.0)
    )
    topic_vec = [topics[i][1] for i in range(num_topics)]
    topic_vec_ids = [topics[i][0] for i in range(num_topics)]
    train_vecs.append(topic_vec)
    train_ids.append(topic_vec_ids)


Topic_List = ['Topic '+str(vis[6][x]) for x in range(num_topics)]



split_df = pd.DataFrame(train_vecs, columns=Topic_List)
CleanDisc_Final = pd.concat([CleanDisc_Final, split_df], axis=1)

CleanDisc_Final['Token_Score_Aged'].plot.kde()

# In[]
CleanDisc_Final['Topic'] = CleanDisc_Final[Topic_List].idxmax(axis=1)
CleanDisc_Final['Topic_Score'] = CleanDisc_Final[Topic_List].max(axis=1)

CleanDisc_Final.to_excel('Post_Analytics.xlsx')

# In[]
Record = CleanDisc_Final.iloc[4]

Rec_Title = Record.Title
Rec_Post_Text = Record['Post Text']
Rec_Text = str(Rec_Title) + ' ' + str(Rec_Post_Text)
Rec_Text_Clean = cleanhtml(Rec_Text)
Rec_Text_Clean_Lower = Rec_Text_Clean.lower()
Rec_Text_Tokens = sent_to_words([Rec_Text_Clean_Lower])
Rec_Text_Tokes_Clean = remove_stopwords(Rec_Text_Tokens)
Rec_Text_Tokes_Clean_Trigram = make_trigrams(Rec_Text_Tokes_Clean)
Rec_Text_Tokes_Clean_Trigram = lemmatization(Rec_Text_Tokes_Clean_Trigram)
Rec_Doc_Map = dictionary.doc2bow(Rec_Text_Tokes_Clean_Trigram[0])
Rec_Top_Topics = model.get_document_topics(Rec_Doc_Map, minimum_probability=0.0)
Rec_Topic_Vec = [Rec_Top_Topics[i][1] for i in range(num_topics)]
Rec_Topic_Probs = pd.DataFrame(Rec_Topic_Vec, index=Topic_List)
Rec_Topic = Rec_Topic_Probs.idxmax(axis=0)[0]
Rec_New_df = Rec_Topic_Probs.T
Rec_New_df['Topic'] = Rec_Topic
Rec_New_df['Tokens'] = Rec_Text_Tokes_Clean_Trigram
Token_vals = list([dictionary.get(x[0]) for x in Rec_Doc_Map])
Rec_New_df['Tokens_Values']  = [Token_vals]
Selected = CleanDisc_Final[CleanDisc_Final.Topic == Rec_Topic]
#Selected = Post_df[Post_df['summary'].str.contains('figure class')]
#Selected = CleanDisc_Final[CleanDisc_Final['Post Text'].str.contains('wwe')]

# In[]
# Sentence Coloring of N Sentences
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

start = 1
end = 15
lda_model = model
# def sentences_chart(lda_model=model, corpus=corpus, start = 0, end = 13):
corp = corpus[start:end]
mycolors = [color for name, color in mcolors.CSS4_COLORS.items()]

fig, axes = plt.subplots(end-start, 1, figsize=(20, (end-start)*0.95), dpi=160)       
axes[0].axis('off')
for i, ax in enumerate(axes):
    if i > 0:
        corp_cur = corp[i-1] 
        topic_percs, wordid_topics, wordid_phivalues = lda_model.get_document_topics(corp_cur,per_word_topics=True)
        word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]    
        ax.text(0.01, 0.5, "Doc " + str(i-1) + ": ", verticalalignment='center',
                fontsize=16, color='black', transform=ax.transAxes, fontweight=700)

        # Draw Rectange
        topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
        ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1, 
                               color=mycolors[topic_percs_sorted[0][0]], linewidth=2))

        word_pos = 0.06
        for j, (word, topics) in enumerate(word_dominanttopic):
            if j < 14:
                ax.text(word_pos, 0.5, word,
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=16, color=mycolors[topics],
                        transform=ax.transAxes, fontweight=700)
                word_pos += .009 * len(word)  # to move the word for the next iter
                ax.axis('off')
        ax.text(word_pos, 0.5, '. . .',
                horizontalalignment='left',
                verticalalignment='center',
                fontsize=16, color='black',
                transform=ax.transAxes)       

plt.subplots_adjust(wspace=0, hspace=0)
plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end-2), fontsize=22, y=0.95, fontweight=700)
plt.tight_layout()
plt.show()

# sentences_chart()    