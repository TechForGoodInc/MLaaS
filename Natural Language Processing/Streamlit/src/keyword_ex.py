import streamlit as st
import pandas as pd
import base64
from keybert import KeyBERT
import flair
import tensorflow_hub
import spacy
from flair.embeddings import TransformerDocumentEmbeddings
from sentence_transformers import SentenceTransformer
import pke
import string
from nltk.corpus import stopwords
from rake_nltk import Metric, Rake
import re
from nltk.corpus import stopwords
import yake


def input():
    no_kw = st.number_input("How many Keywords do you want?")
    n_gram = st.slider("Enter the n-gram", 1, 20)
    raw_text = st.text_area("Your Text")
    return no_kw, n_gram, raw_text

def input_2():
    no_kw = st.number_input("How many Keywords do you want?")
    raw_text = st.text_area("Your Text")
    return no_kw, raw_text

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def sent_trans(raw_text, n_gram, no_kw, model_name): 
    
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
    df['Keyword'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df  

def flair(raw_text, n_gram, no_kw, model_name):
    
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
    df['Keyword'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df 

def use(raw_text, n_gram, no_kw):
    
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
    df['Keyword'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df  

def spacy(raw_text, n_gram, no_kw):
    
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
    df['Keyword'] = kw
    df['Weight'] = weight
    df = df.sort_values(by ='Weight', ascending = False)
    df = df.head(int(no_kw))       
    df = df.reset_index(drop = True)

    return df


def textrank(raw_text, no_kw):    

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



def write_df(df):
    st.write(df)
    tmp_download_link = download_link(df, 'keywords.csv', 'Download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)



def Sentence_Transformer():

    st.subheader("Sentence Transformer")

    options = ['paraphrase-mpnet-base-v2', "paraphrase-MiniLM-L6-v2", "paraphrase-multilingual-mpnet-base-v2", "paraphrase-TinyBERT-L6-v2", "paraphrase-distilroberta-base-v2"]

    st_mod = st.selectbox("Select a Tansformer Model", options) 

    if st_mod == "paraphrase-mpnet-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords", key = "321"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-mpnet-base-v2")
                    write_df(df)

    if st_mod == "paraphrase-MiniLM-L6-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-MiniLM-L6-v2")
                    write_df(df)

    if st_mod == "paraphrase-multilingual-mpnet-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-multilingual-mpnet-base-v2")
                    write_df(df)

    if st_mod == "paraphrase-TinyBERT-L6-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-TinyBERT-L6-v2")
                    write_df(df)

    if st_mod == "paraphrase-distilroberta-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                    df = sent_trans(raw_text, n_gram, no_kw, "paraphrase-distilroberta-base-v2")
                    write_df(df)

    
# ------------------------------------------------------------------------------------------------------------------------- Sentence Transformer ends

def Flair():       
    
    st.subheader("Flair")

    options = ['bert-base-uncased', "roberta-base", "bert-large-uncased", "distilbert-base-uncased-finetuned-sst-2-english", "albert-base-v2"]
    flair_mod = st.selectbox("Select a Tansformer Model", options)
    
    if flair_mod == "bert-base-uncased":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                df = flair(raw_text, n_gram, no_kw, "bert-base-uncased")
                write_df(df)
    
    if flair_mod == "roberta-base":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                df = flair(raw_text, n_gram, no_kw, "roberta-base")
                write_df(df)

    if flair_mod == "bert-large-uncased":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                df = flair(raw_text, n_gram, no_kw, "bert-large-uncased")
                write_df(df)

    if flair_mod == "distilbert-base-uncased-finetuned-sst-2-english":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                df = flair(raw_text, n_gram, no_kw, "distilbert-base-uncased-finetuned-sst-2-english")
                write_df(df)

    if flair_mod == "albert-base-v2":
            no_kw, n_gram, raw_text = input()
            if st.button("Show Keywords"):
                df = flair(raw_text, n_gram, no_kw, "albert-base-v2")
                write_df(df)

# ------------------------------------------------------------------------------------------------------------------------- flair ends

def USE():

    st.subheader("Universal Sentence Encoder (USE)")
    no_kw, n_gram, raw_text = input()
    if st.button("Show Keywords"):
        df = use(raw_text, n_gram, no_kw)
        st.dataframe(df)
        write_df(df)


def spaCy():

    st.subheader("spaCy")
    no_kw, n_gram, raw_text = input()
    if st.button("Show Keywords"):
        df = spacy(raw_text, n_gram, no_kw)
        st.dataframe(df)
        write_df(df)


# ------------------------------------------------------------------------------------------------------------------------- spacy & USE ends


def PKE():
    
    st.subheader(" PKE: Graph-Based Keyphrase Extraction")
    menu = ["TextRank", "SingleRank", "TopicRank", "PositionRank", "MultipartiteRank"]
    choice = st.selectbox("Select", menu)

    if choice == "TextRank":
        st.subheader("TextRank")
        no_kw, raw_text = input_2()
        if st.button("Show Keywords"):
            df = textrank(raw_text, no_kw)
            write_df(df)

    if choice == "SingleRank":
        st.subheader("SingleRank")
        no_kw, raw_text = input_2()
        if st.button("Show Keywords"):
            df = singlerank(raw_text, no_kw)
            write_df(df)
                
    if choice == "TopicRank":
        st.subheader("TopicRank")
        no_kw, raw_text = input_2()
        if st.button("Show Keywords"):
            df = topicrank(raw_text, no_kw)
            write_df(df)

    if choice == "PositionRank":
        st.subheader("PositionRank")
        no_kw, raw_text = input_2()
        if st.button("Show Keywords"):
            df = positionrank(raw_text, no_kw)
            write_df(df)

    if choice == "MultipartiteRank":
        st.subheader("MultipartiteRank")
        no_kw, raw_text = input_2()
        if st.button("Show Keywords"):
            df = multipartiterank(raw_text, no_kw)
            write_df(df)

# ------------------------------------------------------------------------------------------------------------------------- pke ends
            
def rake_():
            
    st.subheader("Rapid Automatic Keyword Extraction")
    no_kw = st.number_input("How many Keywords do you want?")
    min_len = st.slider("Enter Integer for minimum n_gram", 1, 10)
    max_len = st.slider("Enter Integer for maximum n_gram", 1, 20)
    raw_text = st.text_area("Your Text")

    if st.button("Show Keywords"):
            df = rake(raw_text, no_kw)
            write_df(df)

def yake_():

    st.subheader("Light-weight unsupervised automatic keyword extraction")
    no_kw = st.number_input("How many Keywords do you want?")
    max_len = st.slider("Enter Integer for maximum n_gram", 1, 20)
    raw_text = st.text_area("Your Text")

    if st.button("Show Keywords"):
            df = yake(raw_text, no_kw)
            write_df(df)

            
        

