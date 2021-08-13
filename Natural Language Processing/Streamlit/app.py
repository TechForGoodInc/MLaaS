import streamlit as st
import pandas as pd
import base64
import time
from pycaret.nlp import *
import en_core_web_sm
import plotly.figure_factory as ff
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from plotly.graph_objs import *
import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')



def time_():

    with st.empty():
         for seconds in range(60):
             st.write(f"⏳ {seconds} seconds have passed")
             time.sleep(1)
         st.write("✔️ 1 minute over!")

def tii():

    import time

    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)


import os

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


import base64

import streamlit as st
import pandas as pd


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def sent_trans(raw_text, n_gram, no_kw, model_name):
    
    
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    model_ = SentenceTransformer(model_name)
    kw_model = KeyBERT(model=model_)


    kw = []
    weight = []

    a = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(int(n_gram), int(n_gram)), stop_words='english', 
                              use_maxsum=True, nr_candidates=int(no_kw), top_n=int(no_kw))

    for i in list(range(len(a))):
            kw.append(a[i][0])
            weight.append(a[i][1])
            

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df  


def flair(raw_text, n_gram, no_kw, model_name):
    
    
    import flair
    from keybert import KeyBERT
    from flair.embeddings import TransformerDocumentEmbeddings
    
    model_ = TransformerDocumentEmbeddings(model_name)
    kw_model = KeyBERT(model=model_)

    kw = []
    weight = []

    a = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(int(n_gram), int(n_gram)), stop_words='english', 
                              use_maxsum=True, nr_candidates=int(no_kw), top_n=int(no_kw))

    for i in list(range(len(a))):
            kw.append(a[i][0])
            weight.append(a[i][1])
            

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df 


def use(raw_text, n_gram, no_kw):
    

    from keybert import KeyBERT
    import tensorflow_hub

    model_ = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    kw_model = KeyBERT(model=model_)

    kw = []
    weight = []

    a = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(int(n_gram), int(n_gram)), stop_words='english', 
                              use_maxsum=True, nr_candidates=int(no_kw), top_n=int(no_kw))

    for i in list(range(len(a))):
            kw.append(a[i][0])
            weight.append(a[i][1])
            

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df  

def spacy(raw_text, n_gram, no_kw):
    

    from keybert import KeyBERT
    import spacy
    
    nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
    kw_model = KeyBERT(model=nlp)

    kw = []
    weight = []

    a = kw_model.extract_keywords(raw_text, keyphrase_ngram_range=(int(n_gram), int(n_gram)), stop_words='english', 
                              use_maxsum=True, nr_candidates=int(no_kw), top_n=int(no_kw))

    for i in list(range(len(a))):
            kw.append(a[i][0])
            weight.append(a[i][1])
            

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df

def textrank(raw_text, no_kw):    
    
    import pke

    # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # 1. create a TextRank extractor.
    extractor = pke.unsupervised.TextRank()

    # 2. load the content of the document.
    extractor.load_document(input=raw_text,
                            language='en',
                            normalization=None)

    # 3. build the graph representation of the document and rank the words.
    extractor.candidate_weighting(window=100, #the window for connecting two words in the graph
                                  pos=pos,
                                  top_percent=0.33)

    # 4. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=int(no_kw))


    kw = []
    weight = []

    for i in range(len(keyphrases)):
        a = keyphrases[i][0]
        kw.append(a.replace('=', ' ').strip())
        weight.append(keyphrases[i][1])

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)
    
    return df

def singlerank(raw_text, no_kw):    
    
    import pke

    # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # 1. create a SingleRank extractor.
    extractor = pke.unsupervised.SingleRank()

    # 2. load the content of the document.
    extractor.load_document(input=raw_text,
                            language='en',
                            normalization=None)

    # 3. select the longest sequences of nouns and adjectives as candidates.
    extractor.candidate_selection(pos=pos)

    # 4. weight the candidates using the sum of their word's scores that are
    extractor.candidate_weighting(window=100,pos=pos)

    # 4. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=int(no_kw))


    kw = []
    weight = []

    for i in range(len(keyphrases)):
        a = keyphrases[i][0]
        kw.append(a.replace('=', ' ').strip())
        weight.append(keyphrases[i][1])

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)
    
    return df

def topicrank(raw_text, no_kw):    
    
    import pke
    import string
    from nltk.corpus import stopwords

    # 1. create a TopicRank extractor.
    extractor = pke.unsupervised.TopicRank()

    # 2. load the content of the document.
    extractor.load_document(input=raw_text)

    # 3. select the longest sequences of nouns and adjectives, that do
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)

    # 4. build topics by grouping candidates with HAC (average linkage,
    #    threshold of 1/4 of shared stems). Weight the topics using random
    #    walk, and select the first occuring candidate from each topic.
    extractor.candidate_weighting(threshold=0.74, method='average')

    # 5. get the highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=int(no_kw))


    kw = []
    weight = []

    for i in range(len(keyphrases)):
        a = keyphrases[i][0]
        kw.append(a.replace('=', ' ').strip())
        weight.append(keyphrases[i][1])

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)
    
    return df

def positionrank(raw_text, no_kw):    
    
    import pke

    # define the valid Part-of-Speeches to occur in the graph
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # define the grammar for selecting the keyphrase candidates
    grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

    # 1. create a PositionRank extractor.
    extractor = pke.unsupervised.PositionRank()

    # 2. load the content of the document.
    extractor.load_document(input=raw_text,
                            language='en',
                            normalization=None)

    # 3. select the noun phrases up to 3 words as keyphrase candidates.
    extractor.candidate_selection(grammar=grammar,
                                  maximum_word_number=3)

    # 4. weight the candidates using the sum of their word's scores that are
    #    computed using random walk biaised with the position of the words
    #    in the document. In the graph, nodes are words (nouns and
    #    adjectives only) that are connected if they occur in a window of
    #    10 words.
    extractor.candidate_weighting(window=100,
                                  pos=pos)

    # 4. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=int(no_kw))


    kw = []
    weight = []

    for i in range(len(keyphrases)):
        a = keyphrases[i][0]
        kw.append(a.replace('=', ' ').strip())
        weight.append(keyphrases[i][1])

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)
    
    return df

def multipartiterank(raw_text, no_kw):    
    
    import pke
    import string
    from nltk.corpus import stopwords

    # 1. create a MultipartiteRank extractor.
    extractor = pke.unsupervised.MultipartiteRank()

    # 2. load the content of the document.
    extractor.load_document(input=raw_text)

    # 3. select the longest sequences of nouns and adjectives, that do
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'NOUN', 'PROPN', 'ADJ'}
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    extractor.candidate_selection(pos=pos, stoplist=stoplist)

    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.74,
                                  method='average')

    # 4. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=int(no_kw))


    kw = []
    weight = []

    for i in range(len(keyphrases)):
        a = keyphrases[i][0]
        kw.append(a.replace('=', ' ').strip())
        weight.append(keyphrases[i][1])

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)
    
    return df

def rake(raw_text, no_kw, min_len, max_len):
    
    from rake_nltk import Metric, Rake
    import re
    from nltk.corpus import stopwords

    # To use it with a specific language supported by nltk.
    r = Rake(language=None)

    # If you want to provide your own set of stop words and punctuations to
    r = Rake(
        stopwords=stopwords.words('english'),
        punctuations=None
    )

    # If you want to control the metric for ranking. Paper uses d(w)/f(w) as the
    # metric. You can use this API with the following metrics:
    # 1. d(w)/f(w) (Default metric) Ratio of degree of word to its frequency.
    # 2. d(w) Degree of word only.
    # 3. f(w) Frequency of word only.

    r = Rake(ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
    r = Rake(ranking_metric=Metric.WORD_DEGREE)
    r = Rake(ranking_metric=Metric.WORD_FREQUENCY)

    # If you want to control the max or min words in a phrase, for it to be
    # considered for ranking you can initialize a Rake instance as below:

    r = Rake(min_length=int(min_len), max_length=int(max_len))

    r.extract_keywords_from_text(raw_text)
    keyphrases = r.get_ranked_phrases_with_scores()[:int(no_kw)]

    kw = []
    weight = []

    for i in range(len(keyphrases)):
        weight.append(keyphrases[i][0])
        kw.append(re.sub(r'\W+', ' ', keyphrases[i][1]))

    df = pd.DataFrame()    
    df['Kewrord'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df

def yake(raw_text, no_kw, max_len):
    
    import yake

    keyw =[]
    weight = []

    max_ngram_size = max_len
    numOfKeywords = int(no_kw)

    kw_extractor = yake.KeywordExtractor(n=max_ngram_size, top=numOfKeywords, features=None)
    keywords = kw_extractor.extract_keywords(raw_text)

    for kw in keywords:
        keyw.append(kw[0])

    for kw in keywords:
        weight.append(kw[1])

    df = pd.DataFrame()    
    df['Kewrord'] = keyw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False).reset_index(drop = True)
    df = df.head(int(no_kw))
    
    return df

def lda(df, col_name):
                from nltk.corpus import stopwords
                stop_words = stopwords.words('english')
            
                topic_model = setup(data = df, target = col_name, custom_stopwords=stop_words, session_id=21)
                lda = create_model(model='lda', multi_core=True)
                lda_data = assign_model(lda)
                return lda_data

def pycaret_plot(model):
    st.subheader("Topic 0")

    #plot_model(model, plot='wordcloud', topic_num = 'Topic 0', display_format= 'streamlit')
    plot_model(model, plot='frequency', topic_num = 'Topic 0', display_format= 'streamlit')
    plot_model(model, plot='bigram', topic_num = 'Topic 0', display_format= 'streamlit')
    plot_model(model, plot='trigram', topic_num = 'Topic 0', display_format= 'streamlit')
    plot_model(model, plot='distribution', topic_num = 'Topic 0', display_format= 'streamlit')
    plot_model(model, plot='sentiment', topic_num = 'Topic 0', display_format= 'streamlit')

    st.subheader("Topic 1")

    #plot_model(model, plot='wordcloud', topic_num = 'Topic 1', display_format= 'streamlit')
    plot_model(model, plot='frequency', topic_num = 'Topic 1', display_format= 'streamlit')
    plot_model(model, plot='bigram', topic_num = 'Topic 1', display_format= 'streamlit')
    plot_model(model, plot='trigram', topic_num = 'Topic 1', display_format= 'streamlit')
    plot_model(model, plot='distribution', topic_num = 'Topic 1', display_format= 'streamlit')
    plot_model(model, plot='sentiment', topic_num = 'Topic 1', display_format= 'streamlit')

    st.subheader("Topic 2")

    #plot_model(model, plot='wordcloud', topic_num = 'Topic 2', display_format= 'streamlit')
    plot_model(model, plot='frequency', topic_num = 'Topic 2', display_format= 'streamlit')
    plot_model(model, plot='bigram', topic_num = 'Topic 2', display_format= 'streamlit')
    plot_model(model, plot='trigram', topic_num = 'Topic 2', display_format= 'streamlit')
    plot_model(model, plot='distribution', topic_num = 'Topic 2', display_format= 'streamlit')
    plot_model(model, plot='sentiment', topic_num = 'Topic 2', display_format= 'streamlit')

    st.subheader("Topic 3")

    #plot_model(model, plot='wordcloud', topic_num = 'Topic 3', display_format= 'streamlit')
    plot_model(model, plot='frequency', topic_num = 'Topic 3', display_format= 'streamlit')
    plot_model(model, plot='bigram', topic_num = 'Topic 3', display_format= 'streamlit')
    plot_model(model, plot='trigram', topic_num = 'Topic 3', display_format= 'streamlit')
    plot_model(model, plot='distribution', topic_num = 'Topic 3', display_format= 'streamlit')
    plot_model(model, plot='sentiment', topic_num = 'Topic 3', display_format= 'streamlit')

def sa_textprep(df, col_name):

    ## Step 1: Cleaning the text
    import re
    # Define a function to clean the text
    def clean(text):
        # Removes all special characters and numericals leaving the alphabets
        text = re.sub('[^A-Za-z]+', ' ', text) 
        return text

    # Cleaning the text in the review column
    df['Clean Text'] = df[col_name].apply(clean)

    ## Steps 2-4: Tokenization, POS tagging, stopwords removal
    import nltk
    nltk.download('punkt')
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    nltk.download('wordnet')
    from nltk.corpus import wordnet
    # POS tagger dictionary
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}

    def token_stop_pos(text):
        tags = pos_tag(word_tokenize(text))
        newlist = []
        for word, tag in tags:
            if word.lower() not in set(stopwords.words('english')):
                newlist.append(tuple([word, pos_dict.get(tag[0])]))
        return newlist

    df['POS tagged'] = df['Clean Text'].apply(token_stop_pos)

    ## Step 5: Obtaining the stem words
    from nltk.stem import WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()

    def lemmatize(pos_data):
        lemma_rew = " "
        for word, pos in pos_data:
            if not pos: 
                lemma = word
                lemma_rew = lemma_rew + " " + lemma
            else:  
                lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
                lemma_rew = lemma_rew + " " + lemma
        return lemma_rew
        
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    return df

def sa_textblob(df, col_name):

    ## TextBlob in rescue
    from textblob import TextBlob

    # function to calculate subjectivity 
    def getSubjectivity(review):
        return TextBlob(review).sentiment.subjectivity

    # function to calculate polarity
    def getPolarity(review):
        return TextBlob(review).sentiment.polarity

    # function to analyze the reviews
    def analysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'
        
    data_textblob = pd.DataFrame(df[[col_name, 'Lemma']])
    data_textblob['Polarity'] = data_textblob['Lemma'].apply(getPolarity) 
    data_textblob['Analysis'] = data_textblob['Polarity'].apply(analysis)
    data_textblob = data_textblob[[col_name, 'Analysis']]
    data_textblob.columns = [[col_name, 'Sentiment']]
    return data_textblob

def sa_vader(df, col_name):

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    # function to calculate vader sentiment  
    def vadersentimentanalysis(review):
        vs = analyzer.polarity_scores(review)
        return vs['compound']

    # function to analyse 
    def vader_analysis(compound):
        if compound >= 0.5:
            return 'Positive'
        elif compound <= -0.5 :
            return 'Negative'
        else:
            return 'Neutral'
    
    data_vader = pd.DataFrame(df[[col_name, 'Lemma']])
    data_vader['Polarity_Vader'] = data_vader['Lemma'].apply(vadersentimentanalysis)
    data_vader['Analysis'] = data_vader['Polarity_Vader'].apply(vader_analysis)
    data_vader = data_vader[[col_name, 'Analysis']]
    data_vader.columns = [[col_name, 'Sentiment']]
    return data_vader

def sa_swn(df, col_name):

    nltk.download('sentiwordnet')
    from nltk.corpus import sentiwordnet as swn

    def sentiwordnetanalysis(pos_data):
        sentiment = 0
        tokens_count = 0
        for word, pos in pos_data:
            if not pos:
                continue
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            if not lemma:
                continue
            
            synsets = wordnet.synsets(lemma, pos=pos)
            if not synsets:
                continue

            # Take the first sense, the most common
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            sentiment += swn_synset.pos_score() - swn_synset.neg_score()
            tokens_count += 1
            # print(swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score())
        if not tokens_count:
            return 0
        if sentiment>0:
            return "Positive"
        if sentiment==0:
            return "Neutral"
        else:
            return "Negative"

    data_swn = pd.DataFrame(df[[col_name, 'POS tagged']])
    data_swn['Analysis'] = data_swn['POS tagged'].apply(sentiwordnetanalysis)
    data_swn = data_swn[[col_name, 'Analysis']]
    data_swn.columns = [[col_name, 'Sentiment']]
    return data_swn

def spacy_sum(text, no_sent):

    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    from string import punctuation
    from collections import Counter
    from heapq import nlargest
    import nltk
    import heapq
    import re

    nlp = spacy.load('en_core_web_sm') # load the model (English) into spaCy

    doc = nlp(text) #pass the string doc into the nlp function.
    l = len(list(doc.sents)) #number of sentences in the given string
    st.write('Number of sentences in the original text: {}'.format(l) )

    #def generate_summary(text, no_sent):

    doc = nlp(text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')
        
    ## Word frequency 
    word_frequencies = {}  
    for word in nltk.word_tokenize(text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    ## Sentence scoring based on word frequency
    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(int(no_sent), sentence_scores, key=sentence_scores.get)
    summary = ''.join(summary_sentences)
    
    return summary


def lsa_sum(text, no_sent):
    import sumy
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lsa import LsaSummarizer as Summarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    LANGUAGE = "english"
    SENTENCES_COUNT = int(no_sent)
    summary = []

    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))

    ss = []

    for sentence in summarizer(parser.document, SENTENCES_COUNT):
            ss.append(str(sentence))

    summary = ''.join(ss)
    return summary


def luhn_sum(text, no_sent):
    # Import the summarizer
    from sumy.summarizers.luhn import LuhnSummarizer

    # Creating the parser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(text,Tokenizer('english'))

    SENTENCES_COUNT = int(no_sent)

    #  Creating the summarizer
    luhn_summarizer=LuhnSummarizer()
    luhn_summary=luhn_summarizer(parser.document,sentences_count=SENTENCES_COUNT)

    ss = []

    # Printing the summary
    for sentence in luhn_summary:
          ss.append(str(sentence))
            
    summary = ''.join(ss)
    return summary

def kl_sum(text, no_sent): 
    from sumy.summarizers.kl import KLSummarizer
    # Creating the parser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.parsers.plaintext import PlaintextParser
    parser=PlaintextParser.from_string(text,Tokenizer('english'))

    SENTENCES_COUNT = int(no_sent)

    # Instantiating the  KLSummarizer
    kl_summarizer=KLSummarizer()
    kl_summary=kl_summarizer(parser.document,sentences_count=SENTENCES_COUNT)

    ss = []

    # Printing the summary
    for sentence in kl_summary:
        ss.append(str(sentence))

        summary = ''.join(ss)
        return summary


def main():

    st.title("Natural Language Processing")

    menu = ["Keyword Extraction", "Topic Modelling", "Sentiment Analysis", "Text Summarization"]
    choice = st.sidebar.selectbox("Natural Language Processing", menu)

    if choice == "Keyword Extraction":

        menu_main = ["keyBERT", "PKE", "RAKE", "YAKE"]
        choice_main = st.selectbox("Unsupervised Keyword Extraction", menu_main)

        if choice_main == "keyBERT":
            type_task = st.radio("Select from below", ("Sentence Transformer", "Flair", "Universal Sentence Encoder (USE)", "spaCy"))


            if type_task == 'Sentence Transformer':
                    st.subheader("Sentence Transformer")

                    options = ['paraphrase-mpnet-base-v2', "paraphrase-MiniLM-L6-v2", "paraphrase-multilingual-mpnet-base-v2", "paraphrase-TinyBERT-L6-v2", "paraphrase-distilroberta-base-v2"]

                    st_mod = st.selectbox("Select any transformer model from below", options) 

                    if st_mod == "paraphrase-mpnet-base-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords", key = "321"):
                                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-mpnet-base-v2")
                                    st.write(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-mpnet-base-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)

                
                    if st_mod == "paraphrase-MiniLM-L6-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-MiniLM-L6-v2")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-MiniLM-L6-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)


                    if st_mod == "paraphrase-multilingual-mpnet-base-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-multilingual-mpnet-base-v2")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-multilingual-mpnet-base-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if st_mod == "paraphrase-TinyBERT-L6-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-TinyBERT-L6-v2")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-TinyBERT-L6-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if st_mod == "paraphrase-distilroberta-base-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-distilroberta-base-v2")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-distilroberta-base-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
        ## -------------------------------------------------------------------------------------------------------------------------- Sentence Transformer ends
            if type_task == 'Flair':    
                    st.subheader("Flair")
                    options = ['bert-base-uncased', "roberta-base", "bert-large-uncased", "distilbert-base-uncased-finetuned-sst-2-english", "albert-base-v2"]
                    flair_mod = st.selectbox("Select any transformer model from below", options)
                    
                    if flair_mod == "bert-base-uncased":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = flair(raw_text, n_gram, no_kw, "bert-base-uncased")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = flair(raw_text, n_gram, no_kw, "bert-base-uncased")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
                
                    if flair_mod == "roberta-base":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = flair(raw_text, n_gram, no_kw, "roberta-base")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = flair(raw_text, n_gram, no_kw, "roberta-base")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)


                    if flair_mod == "bert-large-uncased":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = flair(raw_text, n_gram, no_kw, "bert-large-uncased")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = flair(raw_text, n_gram, no_kw, "bert-large-uncased")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if flair_mod == "distilbert-base-uncased-finetuned-sst-2-english":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = flair(raw_text, n_gram, no_kw, "distilbert-base-uncased-finetuned-sst-2-english")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = flair(raw_text, n_gram, no_kw, "distilbert-base-uncased-finetuned-sst-2-english")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)

                    if flair_mod == "albert-base-v2":
                            no_kw = st.number_input("How many Keywords do you want?")
                            n_gram = st.number_input("Enter the n-gram")
                            raw_text = st.text_area("Your Text")
                            if st.button("Show Keywords"):
                                    df = flair(raw_text, n_gram, no_kw, "albert-base-v2")
                                    st.dataframe(df)
                            if st.button('Download DataFrame as CSV', key = "123"):
                                df = flair(raw_text, n_gram, no_kw, "albert-base-v2")
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
        ## -------------------------------------------------------------------------------------------------------------------------- flair ends
            if type_task == 'Universal Sentence Encoder (USE)': 
                    st.subheader("Universal Sentence Encoder (USE)")
                    no_kw = st.number_input("How many Keywords do you want?")
                    n_gram = st.number_input("Enter the n-gram")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Keywords"):
                            df = use(raw_text, n_gram, no_kw)
                            st.dataframe(df)
                    if st.button('Download DataFrame as CSV', key = "123"):
                                df = use(raw_text, n_gram, no_kw)
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
        ## -------------------------------------------------------------------------------------------------------------------------- USE ends
            if type_task == "spaCy":
                    st.subheader("spaCy")
                    no_kw = st.number_input("How many Keywords do you want?")
                    n_gram = st.number_input("Enter the n-gram")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Keywords"):
                            df = spacy(raw_text, n_gram, no_kw)
                            st.dataframe(df)
                    if st.button('Download DataFrame as CSV', key = "123"):
                                df = spacy(raw_text, n_gram, no_kw)
                                tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                                st.markdown(tmp_download_link, unsafe_allow_html=True)
    ## -------------------------------------------------------------------------------------------------------------------------- spaCy ends

    ## ---------------------------------------------------------------------------------------------------------------------------------------
    ## -------------------------------------------------------------------------------------------------------------------------- keyBERT ends
    ## ---------------------------------------------------------------------------------------------------------------------------------------
        if choice_main == "PKE":
            st.subheader(" Graph-Based Keyphrase Extraction")
            menu = ["TextRank", "SingleRank", "TopicRank", "PositionRank", "MultipartiteRank"]
            choice = st.selectbox("Select", menu)

            if choice == "TextRank":
                st.subheader("TextRank")
                no_kw = st.number_input("How many Keywords do you want?")
                raw_text = st.text_area("Your Text")
                if st.button("Show Keywords"):
                        df = textrank(raw_text, no_kw)
                        st.dataframe(df)
                if st.button('Download DataFrame as CSV', key = "123"):
                            df = textrank(raw_text, no_kw)
                            tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

            if choice == "SingleRank":
                st.subheader("SingleRank")
                no_kw = st.number_input("How many Keywords do you want?")
                raw_text = st.text_area("Your Text")
                if st.button("Show Keywords"):
                        df = singlerank(raw_text, no_kw)
                        st.dataframe(df)
                if st.button('Download DataFrame as CSV', key = "123"):
                            df = singlerank(raw_text, no_kw)
                            tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                
            if choice == "TopicRank":
                st.subheader("TopicRank")
                no_kw = st.number_input("How many Keywords do you want?")
                raw_text = st.text_area("Your Text")
                if st.button("Show Keywords"):
                        df = topicrank(raw_text, no_kw)
                        st.dataframe(df)
                if st.button('Download DataFrame as CSV', key = "123"):
                            df = topicrank(raw_text, no_kw)
                            tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

            if choice == "PositionRank":
                st.subheader("PositionRank")
                no_kw = st.number_input("How many Keywords do you want?")
                raw_text = st.text_area("Your Text")
                if st.button("Show Keywords"):
                        df = positionrank(raw_text, no_kw)
                        st.dataframe(df)
                if st.button('Download DataFrame as CSV', key = "123"):
                            df = positionrank(raw_text, no_kw)
                            tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

            if choice == "MultipartiteRank":
                st.subheader("MultipartiteRank")
                no_kw = st.number_input("How many Keywords do you want?")
                raw_text = st.text_area("Your Text")
                if st.button("Show Keywords"):
                        df = multipartiterank(raw_text, no_kw)
                        st.dataframe(df)
                if st.button('Download DataFrame as CSV', key = "123"):
                            df = multipartiterank(raw_text, no_kw)
                            tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)

    ## ---------------------------------------------------------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------------------------------------------------ PKE ends
    ## ---------------------------------------------------------------------------------------------------------------------------------------
        if choice_main == "RAKE":
            st.subheader("Rapid Automatic Keyword Extraction")
            no_kw = st.number_input("How many Keywords do you want?")
            min_len = st.number_input("Enter Integer for minimum n_gram")
            max_len = st.number_input("Enter Integer for maximum n_gram")
            raw_text = st.text_area("Your Text")
            if st.button("Show Keywords"):
                    df = rake(raw_text, no_kw, min_len, max_len)
                    st.dataframe(df)
            if st.button('Download DataFrame as CSV', key = "123"):
                        df = rake(raw_text, no_kw, min_len, max_len)
                        tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

    ## ---------------------------------------------------------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------------------------------------------------ RAKE ends
    ## ---------------------------------------------------------------------------------------------------------------------------------------
        if choice_main == "YAKE":
            st.subheader("Light-weight unsupervised automatic keyword extraction")
            no_kw = st.number_input("How many Keywords do you want?")
            max_len = st.number_input("Enter Integer for maximum n_gram")
            raw_text = st.text_area("Your Text")
            if st.button("Show Keywords"):
                    df = yake(raw_text, no_kw, max_len)
                    st.dataframe(df)
            if st.button('Download DataFrame as CSV', key = "123"):
                        df = yake(raw_text, no_kw, max_len)
                        tmp_download_link = download_link(df, 'keywords.csv', 'Click here to download your keywords')
                        t.markdown(tmp_download_link, unsafe_allow_html=True)


    ## ---------------------------------------------------------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------------------------------------------------ KW Extraction ends******************
    ## ---------------------------------------------------------------------------------------------------------------------------------------

    if choice == "Topic Modelling":
        st.subheader("Topic Modelling")
        menu_tm = ["TextBlob, spaCy, Gensim", "PyCaret"]
        choice_tm = st.selectbox("Select", menu_tm)


        if choice_tm == "TextBlob, spaCy, Gensim":
                st.subheader("Latent Dirichlet Allocation ")
                no_top = st.number_input("How many topics do you want?")
                uploaded_file = st.file_uploader("Choose a file")

                col_name = st.text_input("Enter the name of the column for topic modelling and hit enter")
                col_name = str(col_name)
                

                if uploaded_file and col_name is not None:

                    df = pd.read_csv(uploaded_file)
                    df = df[[col_name]]
                    df[col_name] = df[col_name].replace(np.nan, '')

                    st.subheader("Input DatFrame")
                    st.write(df.head())

                    if col_name is not None:

                        def sent_to_words(sentences):
                            for sent in sentences:
                                sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
                                sent = re.sub('\s+', ' ', sent)  # remove newline chars
                                sent = re.sub("\'", "", sent)  # remove single quotes
                                sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
                                yield(sent)  

                        # Convert to list
                        data = df[col_name].values.tolist()
                        data_words = list(sent_to_words(data))


                        # NLTK Stop words
                        from nltk.corpus import stopwords
                        stop_words = stopwords.words('english')
                        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
                        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
                        bigram_mod = gensim.models.phrases.Phraser(bigram)
                        trigram_mod = gensim.models.phrases.Phraser(trigram)

                        # !python3 -m spacy download en  # run in terminal once
                        def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
                            """Remove Stopwords, Form Bigrams, Trigrams and Lemmatization"""
                            texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
                            texts = [bigram_mod[doc] for doc in texts]
                            texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
                            texts_out = []
                            import spacy
                            nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
                            for sent in texts:
                                doc = nlp(" ".join(sent)) 
                                texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
                            # remove stopwords once more after lemmatization
                            texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
                            return texts_out

                        data_ready = process_words(data_words)

                        # Create Dictionary
                        id2word = corpora.Dictionary(data_ready)

                        # Create Corpus: Term Document Frequency
                        corpus = [id2word.doc2bow(text) for text in data_ready]

                        # Build LDA model
                        lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                                   id2word=id2word,
                                                                   num_topics=int(no_top), 
                                                                   random_state=100,
                                                                   update_every=1,
                                                                   chunksize=10,
                                                                   passes=10,
                                                                   alpha='symmetric',
                                                                   iterations=100,
                                                                   per_word_topics=True)

                        def format_topics_sentences(ldamodel=None, corpus=corpus, texts=data):
                            # Init output
                            sent_topics_df = pd.DataFrame()

                            # Get main topic in each document
                            for i, row_list in enumerate(ldamodel[corpus]):
                                row = row_list[0] if ldamodel.per_word_topics else row_list            
                                # print(row)
                                row = sorted(row, key=lambda x: (x[1]), reverse=True)
                                # Get the Dominant topic, Perc Contribution and Keywords for each document
                                for j, (topic_num, prop_topic) in enumerate(row):
                                    if j == 0:  # => dominant topic
                                        wp = ldamodel.show_topic(topic_num)
                                        topic_keywords = ", ".join([word for word, prop in wp])
                                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                                    else:
                                        break
                            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

                            # Add original text to the end of the output
                            contents = pd.Series(texts)
                            sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
                            return(sent_topics_df)

                        topic_model = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_ready)
                        topic_model.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', "Text"]

                        # Format
                        df_dominant_topic = topic_model.reset_index()
                        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

                        # Display setting to show more characters in column
                        pd.options.display.max_colwidth = 100

                        sent_topics_sorteddf_mallet = pd.DataFrame()
                        sent_topics_outdf_grpd = topic_model.groupby('Dominant_Topic')

                        for i, grp in sent_topics_outdf_grpd:
                            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                                                     grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                                                    axis=0)

                        # Reset Index    
                        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

                        # Format
                        sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

                        
                        lda_type = st.radio("Select from below", ("Show Results", "Download Results", "Show Plots"))
                        if lda_type == "Show Results":
                            st.subheader("Resulting DatFrame")
                            st.write(topic_model.head())
                            st.write(sent_topics_sorteddf_mallet.head())
                        if lda_type == "Download Results":
                            tmp_download_link = download_link(topic_model, 'lda.csv', 'Click here to download as CSV')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)
                            tmp_download_link = download_link(sent_topics_sorteddf_mallet, 'lda_keywords.csv', 'Click here to download as CSV')
                            st.markdown(tmp_download_link, unsafe_allow_html=True)


                        ## Plotting starts here ----
                        if lda_type == "Show Plots":
                            plot_type = st.radio("Select from below", ("Document Frequency", "Topic Distribution", "High Frequency Keywords",\
                             "Bigrams & Trigrams", "Sentiment Polarity & Subjectivity", "t-SNE", "WordCloud"))

                            #if plot_type == "pyLDAvis":

                            #    import pyLDAvis
                            #    import pyLDAvis.gensim as gensimvis

                                # Visualize the topics
                                #pyLDAvis.enable_notebook()
                            #    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
                            #    pyLDAvis.show(vis)
                            


                            if plot_type == "Document Frequency":

                                doc_lens = [len(d) for d in df_dominant_topic.Text]

                                import plotly.figure_factory as ff

                                # Add histogram data
                                x1 = doc_lens
                                # Group data together
                                hist_data = [x1]
                                group_labels = ['Token Count']
                                # Create distplot with custom bin_size
                                fig = ff.create_distplot(hist_data, group_labels)
                                fig.update_layout(title_text="Distribution of Document Frequency", xaxis_title="Document Word Count", yaxis_title="Frequency")

                                st.plotly_chart(fig, use_container_width=True)  



                                # Sentence Coloring of N Sentences
                                def topics_per_document(model, corpus, start=0, end=1):
                                    corpus_sel = corpus[start:end]
                                    dominant_topics = []
                                    topic_percentages = []
                                    for i, corp in enumerate(corpus_sel):
                                        topic_percs, wordid_topics, wordid_phivalues = model[corp]
                                        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
                                        dominant_topics.append((i, dominant_topic))
                                        topic_percentages.append(topic_percs)
                                    return(dominant_topics, topic_percentages)

                                dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)            

                                # Distribution of Dominant Topics in Each Document
                                df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
                                dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
                                df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()

                                # Total Topic Distribution by actual weight
                                topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
                                df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()


                                import plotly.express as px
                                fig = px.bar(df_dominant_topic_in_each_doc, x='Dominant_Topic', y='count')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.6)
                                fig.update_layout(title_text="Document Count by Dominant Topic", xaxis_title="Topics", yaxis_title="Number of Documents")
                                st.plotly_chart(fig, use_container_width=True)


                                import plotly.express as px
                                fig = px.bar(df_topic_weightage_by_doc, x='index', y='count')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.8)
                                fig.update_layout(title_text="Frequency by Topic Weightage", xaxis_title="Words", yaxis_title="Number of Documents")
                                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------- Document Frequency Ends
                            
                            if plot_type == "Topic Distribution":

                                topics = lda_model.show_topics(formatted=False)

                                import plotly.figure_factory as ff

                                for i in range(len(topics)):
                                    
                                    df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Dominant_Topic == i, :]
                                    doc_lens = [len(d) for d in df_dominant_topic_sub.Text]

                                    # Add histogram data
                                    x1 = doc_lens
                                    # Group data together
                                    hist_data = [x1]
                                    group_labels = ['Word Count']
                                    # Create distplot with custom bin_size
                                    fig = ff.create_distplot(hist_data, group_labels)
                                    fig.update_layout(title_text="Distribution of Document Word Counts by Dominant Topic: Topic {}".format(i), xaxis_title="Document Word Count", yaxis_title="Number of Documents")

                                    st.plotly_chart(fig, use_container_width=True) 

# ---------------------------------------------------------------------------------------------------------------------------------------------- Topic Distribution Ends

                            if plot_type == "High Frequency Keywords":
                                from collections import Counter
                                topics = lda_model.show_topics(formatted=False)
                                data_flat = [w for w_list in data_ready for w in w_list]
                                counter = Counter(data_flat)

                                out = []
                                for i, topic in topics:
                                    for word, weight in topic:
                                        out.append([word, i , weight, counter[word]])

                                df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])   

                                from collections import Counter
                                topics = lda_model.show_topics(formatted=False)
                                data_flat = [w for w_list in data_ready for w in w_list]
                                counter = Counter(data_flat)

                                out = []
                                for i, topic in topics:
                                    for word, weight in topic:
                                        out.append([word, i , weight, counter[word]])

                                df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])   

                                for i in range(len(topics)):
                                    data = df.loc[df.topic_id==i, :]
                                    data = data.sort_values('word_count', ascending = False)
                                    
                                    import plotly.express as px
                                    fig = px.bar(data, x='word', y='word_count')
                                    fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.8)
                                    fig.update_layout(title_text="Topic Keywords: Topic {}".format(i), xaxis_title="Words", yaxis_title="Count") 
                                    st.plotly_chart(fig, use_container_width=True)   

# ---------------------------------------------------------------------------------------------------------------------------------------------- High Frequency Keywords Ends

                            if plot_type == "Bigrams & Trigrams":

                                from textblob import TextBlob
                                df['polarity'] = df[col_name].apply(lambda x: TextBlob(x).polarity)

                                from nltk.corpus import stopwords
                                stoplist = stopwords.words('english')
                                from sklearn.feature_extraction.text import CountVectorizer
                                                           
                                c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,2))
                                # matrix of ngrams
                                bgrams = c_vec.fit_transform(df[col_name])
                                # count frequency of ngrams
                                count_values = bgrams.toarray().sum(axis=0)
                                # list of ngrams
                                vocab = c_vec.vocabulary_
                                df_bgram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                                            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

                                _bgram = df_bgram.head(50)

                                import plotly.express as px
                                fig = px.bar(_bgram, x='bigram/trigram', y='frequency')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.8)
                                fig.update_layout(title_text="Frequency of Bigrams", xaxis_title="Bigrams", yaxis_title="Frequency")
                                st.plotly_chart(fig, use_container_width=True)


                                from textblob import TextBlob
                                df['polarity'] = df[col_name].apply(lambda x: TextBlob(x).polarity)

                                from nltk.corpus import stopwords
                                stoplist = stopwords.words('english')
                                from sklearn.feature_extraction.text import CountVectorizer
                                                           
                                c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3,3))
                                # matrix of ngrams
                                tgrams = c_vec.fit_transform(df[col_name])
                                # count frequency of ngrams
                                count_values = tgrams.toarray().sum(axis=0)
                                # list of ngrams
                                vocab = c_vec.vocabulary_
                                df_tgram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                                            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

                                _tgram = df_tgram.head(50)

                                import plotly.express as px
                                fig = px.bar(_tgram, x='bigram/trigram', y='frequency')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.8)
                                fig.update_layout(title_text="Frequency of Trigrams", xaxis_title="Trigrams", yaxis_title="Frequency")
                                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------- Trigrams Ends

                            if plot_type == "Sentiment Polarity & Subjectivity":

                                from textblob import TextBlob
                                df['polarity'] = df[col_name].apply(lambda x: TextBlob(x).polarity)
                                
                                from nltk.corpus import stopwords
                                stoplist = stopwords.words('english')
                                from sklearn.feature_extraction.text import CountVectorizer
                                c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
                                # matrix of ngrams
                                ngrams = c_vec.fit_transform(df[col_name])
                                # count frequency of ngrams
                                count_values = ngrams.toarray().sum(axis=0)
                                # list of ngrams
                                vocab = c_vec.vocabulary_
                                df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                                            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

                                df_ngram['polarity'] = df_ngram['bigram/trigram'].apply(lambda x: TextBlob(x).polarity)
                                df_ngram['subjective'] = df_ngram['bigram/trigram'].apply(lambda x: TextBlob(x).subjectivity)
                                pol = df_ngram.head(200)

                                import plotly.express as px
                                fig = px.bar(pol, x='bigram/trigram', y='polarity')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.6)
                                fig.update_layout(title_text="Polarity of Bigrams & Trigrams", xaxis_title="Bigrams & Trigrams", yaxis_title="Sentiment Polarity")
                                st.plotly_chart(fig, use_container_width=True)


                                import plotly.express as px
                                fig = px.bar(pol, x='bigram/trigram', y='subjective')
                                fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                                                  marker_line_width=1.5, opacity=0.6)
                                fig.update_layout(title_text="Subjectivity of Bigrams & Trigrams", xaxis_title="Bigrams & Trigrams", yaxis_title="Sentiment Subjectivity")
                                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------- Sentiment Polarity Ends

                            if plot_type == "t-SNE":

                                import plotly.express as px

                                features = topic_model.loc[:, 'Dominant_Topic': "Perc_Contribution"]
                                tsne = TSNE(n_components=3, random_state=0)
                                projections = tsne.fit_transform(features, )
                                fig = px.scatter_3d(
                                    projections, x=0, y=1, z=2,
                                    color=topic_model.Dominant_Topic, labels={'color': 'Dominant_Topic'}, title="3D TSNE Plot for Topic Models")
                                fig.update_traces(marker_size=8)
                                fig.update_layout(plot_bgcolor='rgba(0,0,0,0.5)')
                                # Plot!
                                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------- t-SNE Ends

                            if plot_type == "WordCloud":

                                topics = lda_model.show_topics(formatted=False)

                                from PIL import Image
                                from wordcloud import WordCloud

                                def create_wordcloud(text):
                                    #stopwords = set(STOPWORDS)
                                    
                                    wc = WordCloud(background_color="black",
                                                  max_words=3000,
                                                  height = 400,
                                                  width =400,
                                                  stopwords=stop_words,
                                                colormap='Blues',
                                                  repeat=True)
                                    wc.generate(str(text))
                                    wc.to_file("wc.png")
                                    
                                    path="wc.png"
                                    im = Image.open(path)

                                    return im
                                    
                                for i in range(len(topics)):
                                    print("Topic {}".format(i))
                                    topic_words = dict(topics[i][1])
                                    image = create_wordcloud(topic_words)
                                    st.image(image, caption='WordCloud: Topic {}'.format(i))
                                   
                                                                                        
# ---------------------------------------------------------------------------------------------------------------------------------------------- WordCloud Ends
        if choice_tm == "PyCaret":
            tm_type = st.radio("Select from below", ("Hierarchical Dirichlet Process", "Random Projections", "Non-Negative Matrix Factorization"))                   


            if tm_type == "Hierarchical Dirichlet Process":   

                st.subheader("Hierarchical Dirichlet Process")
                uploaded_file = st.file_uploader("Choose a file")
                df = pd.read_csv(uploaded_file)

                if uploaded_file is not None:

                    #no_top = st.number_input("How many topics do you want?")
                    col_name = st.text_input("Enter the name of the column for topic modelling and hit enter")
                    col_name = str(col_name)

                    df[col_name] = df[col_name].replace(np.nan, '')
                    df = df[[col_name]]

                    st.subheader("Input DatFrame")
                    st.write(df.head())

                    if col_name is not None:                       

                        from nltk.corpus import stopwords
                        stop_words = stopwords.words('english')

                        pc_tm = setup(data = df, target = col_name, custom_stopwords=stop_words, session_id=21)

                        hdp = create_model(model='hdp', multi_core=True)
                        hdp_data = assign_model(hdp)
                        st.subheader("Resulting DatFrame")
                        st.write(hdp_data.head())
                        tmp_download_link = download_link(hdp_data, 'hdp.csv', 'Click here to download as CSV')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                        plot_model(hdp, plot='topic_distribution', display_format= 'streamlit')
                        #plot_model(model, plot='topic_model', display_format= 'streamlit')
                        plot_model(hdp, plot = 'tsne', display_format= 'streamlit')
                        pycaret_plot(hdp)

            if tm_type == "Random Projections":   

                st.subheader("Random Projections")

                uploaded_file = st.file_uploader("Choose a file")
                df = pd.read_csv(uploaded_file)

                if uploaded_file is not None:

                    #no_top = st.number_input("How many topics do you want?")
                    col_name = st.text_input("Enter the name of the column for topic modelling and hit enter")
                    col_name = str(col_name)

                    df[col_name] = df[col_name].replace(np.nan, '')
                    df = df[[col_name]]

                    st.subheader("Input DatFrame")
                    st.write(df.head())

                    if col_name is not None:                       

                        from nltk.corpus import stopwords
                        stop_words = stopwords.words('english')

                        pc_tm = setup(data = df, target = col_name, custom_stopwords=stop_words, session_id=21)

                        rp = create_model(model='rp', multi_core=True)
                        rp_data = assign_model(rp)
                        st.subheader("Resulting DatFrame")
                        st.write(rp_data.head())
                        tmp_download_link = download_link(rp_data, 'rp.csv', 'Click here to download as CSV')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                        plot_model(rp, plot='topic_distribution', display_format= 'streamlit')
                        #plot_model(model, plot='topic_model', display_format= 'streamlit')
                        plot_model(rp, plot = 'tsne', display_format= 'streamlit')
                        pycaret_plot(rp)

            if tm_type == "Non-Negative Matrix Factorization":   

                st.subheader("Non-Negative Matrix Factorization")
                
                uploaded_file = st.file_uploader("Choose a file")
                df = pd.read_csv(uploaded_file)

                if uploaded_file is not None:

                    #no_top = st.number_input("How many topics do you want?")
                    col_name = st.text_input("Enter the name of the column for topic modelling and hit enter")
                    col_name = str(col_name)

                    df[col_name] = df[col_name].replace(np.nan, '')
                    df = df[[col_name]]

                    st.subheader("Input DatFrame")
                    st.write(df.head())

                    if col_name is not None:                       

                        from nltk.corpus import stopwords
                        stop_words = stopwords.words('english')

                        pc_tm = setup(data = df, target = col_name, custom_stopwords=stop_words, session_id=21)

                        nmf = create_model(model='nmf', multi_core=True)
                        nmf_data = assign_model(nmf)
                        st.subheader("Resulting DatFrame")
                        st.write(nmf_data.head())
                        tmp_download_link = download_link(nmf_data, 'nmf.csv', 'Click here to download as CSV')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)
                        plot_model(nmf, plot='topic_distribution', display_format= 'streamlit')
                        #plot_model(model, plot='topic_model', display_format= 'streamlit')
                        plot_model(nmf, plot = 'tsne', display_format= 'streamlit')
                        pycaret_plot(nmf)


    if choice == "Sentiment Analysis":

        menu_main = ["Unsupervised", "Supervised"]
        choice_main = st.selectbox("Select", menu_main)

        if choice_main == "Unsupervised":

            st.subheader("Rule Based Sentiment Analysis")
            
            uploaded_file = st.file_uploader("Choose a file")

            col_name = st.text_input("Enter the name of the column with text data and hit enter")
            col_name = str(col_name)
                
            if uploaded_file and col_name is not None:

                df = pd.read_csv(uploaded_file)
                df = df[[col_name]]
                df[col_name] = df[col_name].replace('', np.nan)
                df = df.dropna()

                st.subheader("Input DatFrame")
                st.write(df.head())

                st.subheader("Preprocessed Text")
                df = sa_textprep(df, col_name)
                st.write(df)
                tmp_download_link = download_link(df, 'textblob_sentiment.csv', 'Click here to download as CSV')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

                type_task = st.radio("Select from below", ("TextBlob", "Vader", "SentiWordNet"))  

                if type_task ==  "TextBlob": 
                    st.subheader("TextBlob")
                    df_textblob = sa_textblob(df, col_name)
                    tb_counts = df_textblob.Sentiment.value_counts().to_frame().reset_index()
                    tb_counts.columns = ["Sentiment", "Count"]

                    import plotly.express as px
                    fig = px.pie(tb_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'TextBlob Results')
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Resulting DataFrame")
                    st.write(df_textblob)
                    tmp_download_link = download_link(df_textblob, 'textblob_sentiment.csv', 'Click here to download as CSV')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

                if type_task ==  "Vader": 
                    st.subheader("Vader")
                    df_vader = sa_vader(df, col_name)
                    vd_counts = df_vader.Sentiment.value_counts().to_frame().reset_index()
                    vd_counts.columns = ["Sentiment", "Count"]

                    import plotly.express as px
                    fig = px.pie(vd_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Vader Results')
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Resulting DataFrame")
                    st.write(df_vader)
                    tmp_download_link = download_link(df_vader, 'vader_sentiment.csv', 'Click here to download as CSV')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)

                if type_task ==  "SentiWordNet": 
                    st.subheader("SentiWordNet")
                    df_swn = sa_vader(df, col_name)
                    swn_counts = df_swn.Sentiment.value_counts().to_frame().reset_index()
                    swn_counts.columns = ["Sentiment", "Count"]

                    import plotly.express as px
                    fig = px.pie(swn_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'SentiWordNet Results')
                    st.plotly_chart(fig, use_container_width=True)
                    st.subheader("Resulting DataFrame")
                    st.write(df_swn)
                    tmp_download_link = download_link(df_swn, 'swn_sentiment.csv', 'Click here to download as CSV')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)
                    
                    
                    
    if choice == "Text Summarization":

        menu_main = ["Extractive Text Summarization", "Abstractive Text Summarization"]
        choice_main = st.selectbox("Text Summarization", menu_main)

        if choice_main == "Extractive Text Summarization":
            type_task = st.radio("Select from below", ("spaCy", "Latent Semantic Analysis", "Luhn", "KL-Sum"))


            if type_task == 'spaCy':
                    st.subheader("spaCy")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = spacy_sum(raw_text, no_sent)
                        st.info(summary)
                        #('Download Text', key = "123"):
                                
                        tmp_download_link = download_link(summary, 'summary_spacy.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

            if type_task == 'Latent Semantic Analysis':
                    st.subheader("Sumy: Latent Semantic Analysis")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = lsa_sum(raw_text, no_sent)
                        st.info(summary)
                        #('Download Text', key = "123"):
                                
                        tmp_download_link = download_link(summary, 'summary_LSA.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

            if type_task == 'Luhn':
                    st.subheader("Sumy: Luhn")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = luhn_sum(raw_text, no_sent)
                        st.info(summary)
                        #('Download Text', key = "123"):
                                
                        tmp_download_link = download_link(summary, 'summary_Luhn.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)

            if type_task == 'KL-Sum':
                    st.subheader("Sumy: KL-Sum")

                    no_sent = st.number_input("How many sentences would you like in the summarized text?")
                    raw_text = st.text_area("Your Text")
                    if st.button("Show Summary", key = "321"):
                        summary = kl_sum(raw_text, no_sent)
                        st.info(summary)
                                
                        tmp_download_link = download_link(summary, 'summary_KL-Sum.txt', 'Download Text')
                        st.markdown(tmp_download_link, unsafe_allow_html=True)


                    
                       







    
    
if __name__ == '__main__':
    main()
    


