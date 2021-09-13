import streamlit as st
import pandas as pd
import base64

import numpy as np
import re
import sys
import nltk
from textblob import TextBlob
nltk.download('sentiwordnet')
from nltk.corpus import sentiwordnet as swn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.corpus import wordnet

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


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

    df['POS tagged'] = df['POS tagged'].astype('str')
    df['Lemma'] = df['Lemma'].astype('str')
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
    data_textblob.columns = [col_name, 'Sentiment']
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
    data_vader.columns = [col_name, 'Sentiment']
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
    #data_swn['POS tagged'] = data_swn['POS tagged'].astype('str')
    data_swn['Analysis'] = data_swn['POS tagged'].apply(sentiwordnetanalysis)
    data_swn = data_swn[[col_name, 'Analysis']]
    data_swn.columns = [col_name, 'Sentiment']

    return data_swn



def sa_textblob_2(df, col_name):

    df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
    df = df[[col_name]]
    df[col_name] = df[col_name].astype('str')
    def get_polarity(text):
        return TextBlob(text).sentiment.polarity
    df['Polarity'] = df[col_name].apply(get_polarity)

    df.loc[df.Polarity>0,'Sentiment']='Positive'
    df.loc[df.Polarity==0,'Sentiment']='Neutral'
    df.loc[df.Polarity<0,'Sentiment']='Negative'
    return df

def sa_vader_2(df, col_name):

		sid = SentimentIntensityAnalyzer()

		df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
		df = df[[col_name]]
		df[col_name] = df[col_name].astype('str')
		df['scores'] = df[col_name].apply(lambda x: sid.polarity_scores(x))
		df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

		df['Sentiment']=''
		df.loc[df.compound>0,'Sentiment']='Positive'
		df.loc[df.compound==0,'Sentiment']='Neutral'
		df.loc[df.compound<0,'Sentiment']='Negative'
		return df

def sent():

	#st.subheader("Rule Based Sentiment Analysis")
	st.markdown("<h3 style='text-align: center;'>Rule Based Sentiment Analysis</h3>", unsafe_allow_html=True)
	uploaded_file = st.file_uploader("Choose a file")

	while True:
		try:

			df = pd.read_csv(uploaded_file)
			df1 = df.copy()
			st.subheader("First Five Rows of the Input DatFrame")
			st.write(df.head())
	                
			col_name = st.text_input("Enter the name of the column with text data.")
			col_name = str(col_name)

			try:

				df1 = df1[[col_name]]
				df1[col_name] = df1[col_name].replace('', np.nan)
				df1 = df1.dropna()
				st.subheader("Column with Text Data")
				st.write(df1)

				st.subheader("Preprocessed Text")
				df1 = sa_textprep(df1, col_name)
				st.write(df1)
				tmp_download_link = download_link(df1, 'preprocessed.csv', 'Click here to download as CSV')
				st.markdown(tmp_download_link, unsafe_allow_html=True)

				type_task = st.radio("Select from below", ("TextBlob", "Vader"))  

				if type_task ==  "TextBlob": 

							st.subheader("TextBlob")
							
							df_textblob = sa_textblob(df1, col_name)
							df_textblob.columns = [col_name, 'Sentiment']
							tb_counts = df_textblob.Sentiment.value_counts().to_frame().reset_index()
							tb_counts.columns = ["Sentiment", "Count"]

							import plotly.express as px
							fig = px.pie(tb_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: TextBlob Results')
							st.plotly_chart(fig, use_container_width=True)
							st.subheader("Resulting DataFrame")
							st.write(df_textblob)
							tmp_download_link = download_link(df_textblob, 'textblob_sentiment.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)

				if type_task ==  "Vader": 

							st.subheader("Vader")
							
							df_vader = sa_vader(df1, col_name)
							df_vader.columns = [col_name, 'Sentiment']
							vd_counts = df_vader.Sentiment.value_counts().to_frame().reset_index()
							vd_counts.columns = ["Sentiment", "Count"]

							import plotly.express as px
							fig = px.pie(vd_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: Vader Results')
							st.plotly_chart(fig, use_container_width=True)
							st.subheader("Resulting DataFrame")
							st.write(df_vader)
							tmp_download_link = download_link(df_vader, 'vader_sentiment.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)

				#if type_task ==  "SentiWordNet": 

				#			st.subheader("SentiWordNet")
							
				#			df_swn = sa_swn(df1, col_name)
				#			df_swn.columns = [col_name, 'Sentiment']
				#			swn_counts = df_swn.Sentiment.value_counts().to_frame().reset_index()
				#			swn_counts.columns = ["Sentiment", "Count"]

				#			import plotly.express as px
				#			fig = px.pie(swn_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: SentiWordNet Results')
				#			st.plotly_chart(fig, use_container_width=True)
				#			st.subheader("Resulting DataFrame")
				#			st.write(df_swn)
				#			tmp_download_link = download_link(df_swn, 'swn_sentiment.csv', 'Download as CSV')
				#			st.markdown(tmp_download_link, unsafe_allow_html=True)

			except KeyError:        
				st.warning('Please input the column name (case-sensitive).')        
				break
                    
			break
		except ValueError:
			st.warning('Please upload the csv file to proceed.')
			break



