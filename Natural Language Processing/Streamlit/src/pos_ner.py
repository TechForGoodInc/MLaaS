import streamlit as st
import pandas as pd
import base64
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk.chunk import RegexpParser

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

def download_link(object_to_download, download_filename, download_link_text):

    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def nltk(df, col_name):

	df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
	df = df[[col_name]]
	df[col_name] = df[col_name].astype('str')
	df['POS'] = ''
	pattern = 'NP: {<DT>?<JJ>*<NN>}'

	def preprocess(sent):
	    sent = word_tokenize(sent)
	    sent = pos_tag(sent)
	    return sent

	for i in range(len(df)):
		df['POS'][i] = preprocess(df[col_name][i])
		cp = RegexpParser(pattern)
		cs = cp.parse(df['POS'][i])
		df['POS'][i] = tree2conlltags(cs)

	df.POS = df.POS.astype(str)
	df.columns = [col_name,'(Word, POS, NER)']
	return df

def spacy_posner(df, col_name):

	df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
	df = df[[col_name]]
	df[col_name] = df[col_name].astype('str')

	pos_ner = []

	for i in range(len(df)):
	    doc = nlp(df[col_name][i])
	    pos_ner.append([(X, X.tag_, X.ent_iob_, X.ent_type_) for X in doc])
    
	df['Word, POS, NER, NE Type'] = pos_ner

	df['Word, POS, NER, NE Type'] = df['Word, POS, NER, NE Type'].astype(str)
	return df

def spacy_posner1(df, col_name):

	df.drop_duplicates(subset = col_name, keep = 'first', inplace = True)
	df = df[[col_name]]
	df[col_name] = df[col_name].astype('str')

	lemma = []
	pos = []
	ner = []
	ner_type = []

	for i in range(len(df)):
	    doc = nlp(df[col_name][i])
	    lemma.append([(X, X.lemma_) for X in doc])
	    pos.append([(X, X.tag_) for X in doc])
	    ner.append([(X, X.ent_iob_) for X in doc])
	    ner_type.append([(X, X.ent_type_) for X in doc])
    
	df['Word, Lemma'] = lemma
	df['Word, POS'] = pos
	df['Word, IOB'] = ner
	df['Word, NER Type'] = ner_type

	df['Word, Lemma'] = df['Word, Lemma'].astype(str)
	df['Word, POS'] = df['Word, POS'].astype(str)
	df['Word, IOB'] = df['Word, IOB'].astype(str)
	df['Word, NER Type'] = df['Word, NER Type'].astype(str)
	return df

def posner():

	#st.subheader("POS Tagging & Named Entity Recognition")
	st.markdown("<h3 style='text-align: center;'>POS Tagging & Named Entity Recognition</h3>", unsafe_allow_html=True)
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

				type_task = st.radio("Select from below", ("NLTK", "spaCy"))

				if type_task ==  "NLTK": 

							st.subheader("NLTK: Noun Phrase Chunking")
							st.markdown('Our chunk pattern consists of one rule, that a noun phrase, NP, should be formed whenever the chunker finds an optional determiner, DT, followed by any number of adjectives, JJ, and then a noun, NN.')
							st.info("pattern = 'NP: {<DT>?<JJ>*<NN>}'")
							df_nltk = nltk(df, col_name)

							st.subheader("Resulting DataFrame")
							st.write(df_nltk)
							tmp_download_link = download_link(df_nltk, 'nltk_pos_ner.csv', 'Download as CSV')
							st.markdown(tmp_download_link, unsafe_allow_html=True)

				if type_task ==  "spaCy": 

							st.subheader("spaCy")

							menu_main = ["POS Tagging & NER", "Visualize NER"]
							choice_main = st.selectbox("Select", menu_main)

							if choice_main == "POS Tagging & NER":

								df_spacy = spacy_posner(df, col_name)
								df_spacy1 = spacy_posner1(df, col_name)

								st.subheader("Resulting DataFrames")
								st.write(df_spacy)
								tmp_download_link = download_link(df_spacy, 'spacy_pos_ner.csv', 'Download as CSV')
								st.markdown(tmp_download_link, unsafe_allow_html=True)

								st.write(df_spacy1)
								tmp_download_link = download_link(df_spacy, 'spacy_pos_ner.csv', 'Download as CSV')
								st.markdown(tmp_download_link, unsafe_allow_html=True)

							if choice_main == "Visualize NER":

								st.subheader("Visualize Entities")

								raw_text = st.text_area("Your Text")

								if st.button("Show Results", key = "321"):
									import spacy
									from spacy_streamlit import visualize_ner
									import spacy_streamlit

									nlp = spacy.load("en_core_web_sm")
									doc = nlp(raw_text)
									visualize_ner(doc, labels=nlp.get_pipe("ner").labels)


			except KeyError:        
				st.warning('Please input the column name (case-sensitive).')        
				break
                    
			break
		except ValueError:
			st.warning('Please upload the csv file to proceed.')
			break