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


def sa_textblob(df, col_name):

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

def sa_vader(df, col_name):

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

	st.subheader("Rule Based Sentiment Analysis")
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

				type_task = st.radio("Select from below", ("TextBlob", "Vader"))

				if type_task ==  "TextBlob": 

							st.subheader("TextBlob")
							df_textblob = sa_textblob(df, col_name)
							tb_counts = df_textblob.Sentiment.value_counts().to_frame().reset_index()
							tb_counts.columns = ["Sentiment", "Count"]

							import plotly.express as px
							fig = px.pie(tb_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: TextBlob Results', opacity = 0.8)
							st.plotly_chart(fig, use_container_width=True)
							st.subheader("Resulting DataFrame")
							st.write(df_textblob)
							tmp_download_link = download_link(df_textblob, 'textblob_sentiment.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)

				if type_task ==  "Vader": 

							st.subheader("Vader")
							df_vader = sa_vader(df, col_name)
							tb_counts = df_vader.Sentiment.value_counts().to_frame().reset_index()
							tb_counts.columns = ["Sentiment", "Count"]

							import plotly.express as px
							fig = px.pie(tb_counts, values='Count', names='Sentiment', color_discrete_sequence=px.colors.sequential.RdBu, title= 'Sentiment Categories: Vader Results', opacity = 0.8)
							st.plotly_chart(fig, use_container_width=True)
							st.subheader("Resulting DataFrame")
							st.write(df_vader)
							tmp_download_link = download_link(df_vader, 'vader_sentiment.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)



			except KeyError:        
				st.warning('Please input the column name (case-sensitive).')        
				break
                    
			break
		except ValueError:
			st.warning('Please upload the csv file to proceed.')
			break




