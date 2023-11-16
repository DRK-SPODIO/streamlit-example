#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 08:27:54 2022

@author: spodionew
"""

#!/usr/bin/env python
# coding: utf-8

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import pandas as pd
import itertools


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




def Get_Data():
    return pd.read_excel('Doc2Vec Results.xlsx')

# Depricated, See CreateSummary()
# def read_article(Topic):
#     file = Get_Data()
#     filedata = file[file.Topic == Topic]
#     filedata = filedata.head(15)
#     filedata = filedata['summary'].tolist()
#     filedata = '. '.join(filedata)
#     # Join posts into single large article
#     article = filedata[1].split(". ")
#     sentences = []

#     for sentence in article:
#         print(sentence)
#         sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
#     sentences.pop() 
    
#     return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

# Depricated, See Create Summary()
# def generate_summary(Topic, top_n=5):
#     stop_words = STOPWORDS
#     summarize_text = []

#     # Step 1 - Read text anc split it
#     sentences =  read_article(Topic)

#     # Step 2 - Generate Similary Martix across sentences
#     sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

#     # Step 3 - Rank sentences in similarity martix
#     sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
#     scores = nx.pagerank(sentence_similarity_graph)

#     # Step 4 - Sort the rank and pick top sentences
#     ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
#     print("Indexes of top ranked_sentence order are ", ranked_sentence)    

#     for i in range(top_n):
#       summarize_text.append(" ".join(ranked_sentence[i][1]))

#     # Step 5 - Offcourse, output the summarize texr
#     print("Summarize Text: \n", ". ".join(summarize_text))


# In[]
# Read in Data
file = Get_Data()
# In[]
from lxml import etree


# Function to remove HTML and Script elements from RSS Posts
def cleanhtml(raw_html):
    response = etree.HTML(text=raw_html)
    return response.xpath('string(.)')

ExceptionLog = []
def CreateSummary(Topic, top_n = 2):
    # For some topics, the simularity graph fails to converge raising a
    # PowerIterationFAileConvergence Exception.  We handle this exception
    # by returning a "No Summary." for that Topic.
    try:
        # Initalize Vars
        summarize_text = []
        # Filter all posts to just those within the topic
        filedata = file[file.Topic == Topic]
        # Filter out posts with no Summary and sort by Relivence-Age Score
        filedata = filedata[filedata.summary != 'No Summary']
        filedata = filedata.sort_values(by='Rel_Age_Score',axis=0,ascending=False)
        # Remove HTML elements & strip leading/trailing whitespace
        filedata.summary = [cleanhtml(x) for x in filedata.summary]
        filedata.summary = [x.strip() for x in filedata.summary]
        # Limit to top # of posts (by Relivence-Age Score)
        summary = filedata.head(20)
        # Join Post Summaries into single 'article'
        summary = summary['summary'].tolist()
        summary = '.  '.join(summary)
        # Eliminate Doulble periods
        summary = summary.replace('..','.')
        # Seperate article into sentences and strip out whitespace characters
        article = summary.split(".  ")
        article = [' '.join(x.split()) for x in article]
        
        # Tokenize Sentences, remove short & duplicate sentences
        sentences = []
        for sentence in article:
            sentences.append(sentence.replace("[^a-zA-Z]", " ").strip().split(" "))
        sentences.pop() 
        
        # Pre-processing Tokens; remove short and duplicate Sentences
        sentences = [x for x in sentences if len(x) > 4]
        sentences.sort()
        sentences = list(sent for sent,_ in itertools.groupby(sentences))
        # Remove HTTP links (token-wise)
        sentences = [[token for token in sentence if 'http' not in token] for sentence in sentences]
        
        # Generate Similary Martix across sentences
        sentence_similarity_martix = build_similarity_matrix(sentences, STOPWORDS)
        
        # Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph, max_iter=1000)
        
        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
        #print("Indexes of top ranked_sentence order are ", ranked_sentence)    
        
        for i in range(top_n):
          summarize_text.append(" ".join(ranked_sentence[i][1]))
        
        # Step 5 - Offcourse, output the summarize texr
        print(f"Summarize Text for {Topic}: \n", ".  ".join(summarize_text).strip()+'.')
        return [Topic,len(article),".  ".join(summarize_text).strip()+'.']
    except Exception as e:
        ExceptionLog.append(e)
        print(f"Summarize Text for {Topic}: \n", 'No Summary.')
        return [Topic,len(article),'No Summary.']
    

# Set a Time Bracket;  All posts from PostDate to Today would be included.
PostDate = '6/1/2022'  # [Date, None]  Setting to None ignores this filter
#PostDate = None  # [Date, None]  Setting to None ignores this filter

#Filter to Topics that have greater than 20 posts in the timeframe and have summaries.
# Exclude posts with "No Summary"
Topics = file[file.summary != 'No Summary']
# Check for Date Bracket
if PostDate is not None:
    Topics = Topics[Topics.published > PostDate]

#Get the Post Count for each Topic
Topics = Topics.groupby(by='Topic').count()
# Exclude Topics with few posts (20 in this case)
# In the Groupby.count, all columns = the post count, so Age is not specific;
Topics = Topics[Topics.Age > 19]
# Sort Topics by number of Posts and restrict to top 10
Topics = Topics.sort_values(by='Age',ascending=False)
# Limit to the top Topics, if there are more than this number
Topics = Topics.head(50)
# Generate the list of topics to find summaries for.
Topics = Topics.index.tolist()

Topic_Summaries = []
for Topic in Topics:
    Topic_Summaries.append(CreateSummary(Topic,top_n = 2))

# In[]
# DEBUG/DEV
import matplotlib.pyplot as plt
top_n = 2
Topic = 'Topic 9'
# For some topics, the simularity graph fails to converge raising a
# PowerIterationFAileConvergence Exception.  We handle this exception
# by returning a "No Summary." for that Topic.
try:
    # Initalize Vars
    summarize_text = []
    # Filter all posts to just those within the topic
    filedata = file[file.Topic == Topic]
    # Filter out posts with no Summary and sort by Relivence-Age Score
    filedata = filedata[filedata.summary != 'No Summary']
    filedata = filedata.sort_values(by='Rel_Age_Score',axis=0,ascending=False)
    # Remove HTML elements & strip leading/trailing whitespace
    filedata.summary = [cleanhtml(x) for x in filedata.summary]
    filedata.summary = [x.strip() for x in filedata.summary]
    # Limit to top # of posts (by Relivence-Age Score)
    summary = filedata.head(25)
    #summary = filedata
    # Join Post Summaries into single 'article'
    summary = summary['summary'].tolist()
    summary = '.  '.join(summary)
    # Eliminate Doulble periods
    summary = summary.replace('..','.')
    # Seperate article into sentences and strip out whitespace characters
    import spacy
    import en_core_web_trf
    Sentenizer = spacy.load('en_core_web_trf')
    #article = summary.split(".  ")
    articleNLP = Sentenizer(summary)
    article = [sentence for sentence in articleNLP.sents]
    article = [' '.join(x.split()) for x in article]
    
    # Tokenize Sentences, remove short & duplicate sentences
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").strip().split(" "))
    sentences.pop() 
    
    # Pre-processing Tokens; remove short and duplicate Sentences
    sentences = [x for x in sentences if len(x) > 4]
    sentences.sort()
    sentences = list(sent for sent,_ in itertools.groupby(sentences))
    # Remove HTTP links (token-wise)
    sentences = [[token for token in sentence if 'http' not in token] for sentence in sentences]
    
    # Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, STOPWORDS)
    
    # Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph, max_iter=1000)
    nx.draw(sentence_similarity_graph)
    plt.show()
    nx.draw_kamada_kawai(sentence_similarity_graph)
    
    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    # Step 5 - Offcourse, output the summarize texr
    print(f"Summarize Text for {Topic}: \n", ".  ".join(summarize_text).strip()+'.')
    #return [Topic,len(article),".  ".join(summarize_text).strip()+'.']
except Exception as e:
    ExceptionLog.append(e)
    print(e)
    print(f"Summarize Text for {Topic}: \n", 'No Summary.')
    #return [Topic,len(article),'No Summary.']

# In[]
